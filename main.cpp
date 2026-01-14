/*
 * Fractal City - L-System with OpenMP Parallelization
 * 
 * Compile WITHOUT OpenMP on macOS (recommended, works out of the box):
 *   g++ -std=c++17 -I/opt/homebrew/include -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -L/opt/homebrew/lib main.cpp -lsfml-graphics -lsfml-window -lsfml-system -o fractal_city
 * 
 * Compile WITH OpenMP on Linux:
 *   g++ -std=c++17 -fopenmp main.cpp -lsfml-graphics -lsfml-window -lsfml-system -o fractal_city
 * 
 * Compile WITH OpenMP on macOS (requires: brew install libomp):
 *   g++ -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/include -I/opt/homebrew/opt/libomp/include -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib -lomp main.cpp -lsfml-graphics -lsfml-window -lsfml-system -o fractal_city
 * 
 * The code uses fork-join parallelization with openmp making it way faster:
 * Sequential: L-system string generation (iterations), turtle state tracking
 * Parallel: Character replacement in L-system, building geometry generation
 * 
 * Note: Code automatically falls back to sequential execution if OpenMP is not available.
 */

#include <SFML/Graphics.hpp>
#include <optional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif


sf::Color hsvToRgb(float h, float s, float v)
{
    // keep hue in [0, 360)
    h = std::fmod(h, 360.f);
    if (h < 0) h += 360.f;

    // clamp s, v to [0, 1]
    if (s < 0.f) s = 0.f;
    if (s > 1.f) s = 1.f;
    if (v < 0.f) v = 0.f;
    if (v > 1.f) v = 1.f;

    float c = v * s;
    float x = c * (1.f - std::fabs(std::fmod(h / 60.f, 2.f) - 1.f));
    float m = v - c;

    float rPrime = 0.f, gPrime = 0.f, bPrime = 0.f;

    if (0.f <= h && h < 60.f) {
        rPrime = c; gPrime = x; bPrime = 0.f;
    } else if (60.f <= h && h < 120.f) {
        rPrime = x; gPrime = c; bPrime = 0.f;
    } else if (120.f <= h && h < 180.f) {
        rPrime = 0.f; gPrime = c; bPrime = x;
    } else if (180.f <= h && h < 240.f) {
        rPrime = 0.f; gPrime = x; bPrime = c;
    } else if (240.f <= h && h < 300.f) {
        rPrime = x; gPrime = 0.f; bPrime = c;
    } else { // 300–360
        rPrime = c; gPrime = 0.f; bPrime = x;
    }

    unsigned char r = static_cast<unsigned char>((rPrime + m) * 255.f);
    unsigned char g = static_cast<unsigned char>((gPrime + m) * 255.f);
    unsigned char b = static_cast<unsigned char>((bPrime + m) * 255.f);

    return sf::Color(r, g, b);
}

// L-system string generation with OpenMP parallelization
std::string generateLSystem(const std::string& axiom,
                            const std::unordered_map<char, std::string>& rules,
                            int iterations)
{
    std::string current = axiom;
    
    // Safety limit to prevent excessive memory usage
    const std::size_t MAX_STRING_SIZE = 50'000'000; // 50 million chars max

    for (int i = 0; i < iterations; ++i) {
        // Check if we're getting too large
        if (current.size() > MAX_STRING_SIZE) {
            std::cerr << "Warning: L-system string exceeded size limit, truncating at iteration " << i << "\n";
            break;
        }
        
        std::string next;
        // More conservative reserve - rules can expand significantly
        // Estimate: average 3-4 chars per char for most L-systems
        std::size_t estimatedSize = current.size() * 4;
        if (estimatedSize > MAX_STRING_SIZE) {
            estimatedSize = MAX_STRING_SIZE;
        }
        // Don't exceed string's max_size
        estimatedSize = std::min(estimatedSize, next.max_size());
        next.reserve(estimatedSize);
        
        // Parallelize character replacement (fork-join pattern)
        // Sequential: iteration loop, Parallel: character processing
        const std::size_t strLen = current.size();
        std::vector<std::string> chunks;
        
        #ifdef _OPENMP
        // Only parallelize if string is large enough to benefit
        // For small strings, overhead isn't worth it
        if (strLen > 1000) {
            const int numThreads = omp_get_max_threads();
            const std::size_t chunkSize = std::max(std::size_t(1), strLen / numThreads);
            
            chunks.resize(numThreads);
            
            // OpenMP parallel section - character replacement across threads
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < numThreads; ++t) {
                std::size_t start = t * chunkSize;
                std::size_t end = (t == numThreads - 1) ? strLen : (t + 1) * chunkSize;
                
                std::string localChunk;
                // More conservative reserve 
                //rules can expand characters significantly
                // Estimate: average 4 chars per char (some rules expand a lot)
                localChunk.reserve((end - start) * 4);
                
                for (std::size_t j = start; j < end; ++j) {
                    char c = current[j];
                    if (rules.count(c)) {
                        const std::string& replacement = rules.at(c);
                        localChunk += replacement;
                    } else {
                        localChunk += c;
                    }
                }
                
                chunks[t] = std::move(localChunk);
            }
            
            // Sequential join: combine chunks from all threads
            // Pre-calculate total size for better performance
            std::size_t totalSize = 0;
            for (const auto& chunk : chunks) {
                totalSize += chunk.size();
            }
            // Safety check: don't exceed string max_size
            if (totalSize > next.max_size()) {
                std::cerr << "Error: L-system string would exceed maximum size\n";
                return current; // Return current state instead of crashing
            }
            next.reserve(std::min(totalSize, next.max_size()));
            
            for (const auto& chunk : chunks) {
                next += chunk;
            }
        } else {
            // Sequential for small strings
            for (char c : current) {
                if (rules.count(c))
                    next += rules.at(c);
                else
                    next += c;
            }
        }
        #else
        // Sequential fallback (OpenMP not available)
        for (char c : current) {
            if (rules.count(c))
                next += rules.at(c);
            else
                next += c;
        }
        #endif

        current = next;
    }

    return current;
}

// Geometry building from L-system string
struct TurtleState {
    sf::Vector2f position;
    float angleDeg;
    int depth;
};

struct SceneGeometry {
    sf::VertexArray roads;
    sf::VertexArray buildings;
    sf::VertexArray particles;
    sf::VertexArray shadows;
};

// Simple random number generator 
struct SimpleRNG {
    unsigned int seed;
    SimpleRNG(unsigned int s) : seed(s) {}
    float next() {
        seed = seed * 1103515245 + 12345;
        return (seed & 0x7FFFFFFF) / 2147483648.0f;
    }
};

// Structure to hold segment data for parallel processing
struct SegmentData {
    sf::Vector2f oldPos;
    sf::Vector2f newPos;
    float hue;
    float sat;
    int depth;
    std::size_t segmentIndex;
};

SceneGeometry buildGeometry(const std::string& sequence,
                            float angleDeg,
                            float stepLength,
                            sf::Vector2f startPos,
                            std::size_t maxSymbols,
                            float baseHue,
                            float hueStep,
                            bool addBuildings,
                            float time,
                            float windStrength)
{
    SceneGeometry geom{
        sf::VertexArray(sf::PrimitiveType::Lines),
        sf::VertexArray(sf::PrimitiveType::Lines),
        sf::VertexArray(sf::PrimitiveType::Points),
        sf::VertexArray(sf::PrimitiveType::Triangles)
    };

    // phase 1
    // Sequential turtle state tracking (fork-join pattern: sequential phase)
    std::vector<SegmentData> segments;
    segments.reserve(maxSymbols / 2); // rough estimate

    TurtleState turtle;
    turtle.position = startPos;
    turtle.angleDeg = 0.f; // facing right
    turtle.depth = 0;

    std::vector<TurtleState> stack;

    std::size_t used = 0;
    std::size_t segmentIndex = 0; // how many F segments we've drawn

    for (char c : sequence) {
        if (used >= maxSymbols) break;

        switch (c) {
            case 'F': {
                used++;

                // Add subtle wind effect that makes things flow
                float windOffset = std::sin(time * 0.5f + segmentIndex * 0.1f) * windStrength;
                float currentAngle = turtle.angleDeg + windOffset;
                
                float rad = currentAngle * 3.14159265f / 180.f;
                sf::Vector2f oldPos = turtle.position;
                
                // RNG seeded by segment index for consistent but varied results
                SimpleRNG rng(static_cast<unsigned int>(segmentIndex * 7919 + 12345));
                
                // Add slight wobble for organic feel
                float wobble = rng.next() * 0.3f - 0.15f;
                float step = stepLength * (1.f + wobble * 0.1f);
                
                sf::Vector2f newPos{
                    turtle.position.x + step * std::cos(rad),
                    turtle.position.y + step * std::sin(rad)
                };

                float hue = baseHue + static_cast<float>(segmentIndex) * hueStep;
                float sat = 0.85f + rng.next() * 0.15f;

                // Store segment data for parallel processing
                SegmentData seg;
                seg.oldPos = oldPos;
                seg.newPos = newPos;
                seg.hue = hue;
                seg.sat = sat;
                seg.depth = turtle.depth;
                seg.segmentIndex = segmentIndex;
                segments.push_back(seg);

                // Roads can be added immediately (fast)
                sf::Color roadColor = hsvToRgb(hue, sat, 1.f);
                sf::Vertex v1;
                v1.position = oldPos;
                v1.color = roadColor;
                sf::Vertex v2;
                v2.position = newPos;
                v2.color = roadColor;
                geom.roads.append(v1);
                geom.roads.append(v2);

                turtle.position = newPos;
                segmentIndex++;
                break;
            }
            case '+':
                used++;
                turtle.angleDeg += angleDeg;
                break;
            case '-':
                used++;
                turtle.angleDeg -= angleDeg;
                break;
            case '[':
                used++;
                stack.push_back(turtle);
                turtle.depth++; // go one level deeper
                break;
            case ']':
                used++;
                if (!stack.empty()) {
                    turtle = stack.back();
                    stack.pop_back();
                    // depth restored from saved state
                }
                break;
            default:
                // 'A', 'B', etc: non-drawing symbols
                break;
        }
    }

    // PHASE 2: Parallel building generation (fork-join pattern: parallel phase)
    if (addBuildings && !segments.empty()) {
        #ifdef _OPENMP
        // OpenMP PARALLEL: Building geometry generation across multiple threads
        const std::size_t numSegments = segments.size();
        const int numThreads = omp_get_max_threads();
        std::vector<std::vector<sf::Vertex>> buildingVerts(numThreads);
        std::vector<std::vector<sf::Vertex>> shadowVerts(numThreads);
        std::vector<std::vector<sf::Vertex>> particleVerts(numThreads);
        
        // Fork: Start parallel region 
        //each thread processes a subset of segments
        #pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            std::vector<sf::Vertex>& localBuildings = buildingVerts[threadId];
            std::vector<sf::Vertex>& localShadows = shadowVerts[threadId];
            std::vector<sf::Vertex>& localParticles = particleVerts[threadId];
            
            // Parallel loop: segments distributed across threads
            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < numSegments; ++i) {
                const SegmentData& seg = segments[i];
                
                sf::Vector2f mid{
                    (seg.oldPos.x + seg.newPos.x) * 0.5f,
                    (seg.oldPos.y + seg.newPos.y) * 0.5f
                };

                // RNG for this segment (deterministic based on index)
                SimpleRNG rng(static_cast<unsigned int>(seg.segmentIndex * 7919 + 12345));
                
                // Height grows with depth + some randomness
                float baseHeight = 6.f;
                float height = baseHeight + static_cast<float>(seg.depth) * 4.f;
                height += rng.next() * 8.f; // random variation
                
                // Building width varies
                float width = 3.f + rng.next() * 4.f;
                
                // Isometric 3D effect - draw building as a box
                float isoX = width * 0.5f;
                float isoY = width * 0.25f; // isometric perspective
                
                // Building color with depth variation
                float buildingHue = seg.hue + 20.f; // slight shift
                float buildingVal = 0.5f + static_cast<float>(seg.depth) * 0.1f;
                if (buildingVal > 1.f) buildingVal = 1.f;
                sf::Color buildingColor = hsvToRgb(buildingHue, 0.7f, buildingVal);
                sf::Color buildingColorDark = hsvToRgb(buildingHue, 0.7f, buildingVal * 0.6f);
                
                // Front face (vertical)
                // 8 vertices for 4 lines
                sf::Vertex v;
                v.position = sf::Vector2f{mid.x - isoX, mid.y};
                v.color = buildingColor;
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x - isoX, mid.y};
                localBuildings.push_back(v);
                
                // Top face (isometric) - 6 vertices for triangle
                v.color = buildingColorDark;
                v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - height - isoY};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - height - isoY};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x - isoX + isoY, mid.y - height - isoY};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
                localBuildings.push_back(v);
                
                // Side face 
                // 6 vertices for triangle
                v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - height - isoY};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - isoY};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - isoY};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y};
                localBuildings.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
                localBuildings.push_back(v);
                
                // Shadow 3 vertices for triangle
                v.color = sf::Color(0, 0, 0, 80);
                v.position = sf::Vector2f{mid.x - isoX + isoY, mid.y - isoY};
                localShadows.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - isoY};
                localShadows.push_back(v);
                v.position = sf::Vector2f{mid.x + isoX, mid.y};
                localShadows.push_back(v);
                
                // Add particles/floating elements around buildings
                if (rng.next() > 0.7f) {
                    float particleX = mid.x + (rng.next() - 0.5f) * 20.f;
                    float particleY = mid.y - height + (rng.next() - 0.5f) * 10.f;
                    float particleHue = seg.hue + rng.next() * 60.f;
                    v.position = sf::Vector2f{particleX, particleY};
                    v.color = hsvToRgb(particleHue, 0.8f, 0.9f);
                    localParticles.push_back(v);
                }
            } // End of parallel for loop
        } // End of OpenMP parallel region (join point)
        
        // Sequential join: combine all thread-local results from parallel computation
        for (const auto& verts : buildingVerts) {
            for (const auto& v : verts) {
                geom.buildings.append(v);
            }
        }
        for (const auto& verts : shadowVerts) {
            for (const auto& v : verts) {
                geom.shadows.append(v);
            }
        }
        for (const auto& verts : particleVerts) {
            for (const auto& v : verts) {
                geom.particles.append(v);
            }
        }
        #else
      
        // Sequential fallback
        for (const SegmentData& seg : segments) {
            sf::Vector2f mid{
                (seg.oldPos.x + seg.newPos.x) * 0.5f,
                (seg.oldPos.y + seg.newPos.y) * 0.5f
            };

            SimpleRNG rng(static_cast<unsigned int>(seg.segmentIndex * 7919 + 12345));
            
            float baseHeight = 6.f;
            float height = baseHeight + static_cast<float>(seg.depth) * 4.f;
            height += rng.next() * 8.f;
            
            float width = 3.f + rng.next() * 4.f;
            float isoX = width * 0.5f;
            float isoY = width * 0.25f;
            
            float buildingHue = seg.hue + 20.f;
            float buildingVal = 0.5f + static_cast<float>(seg.depth) * 0.1f;
            if (buildingVal > 1.f) buildingVal = 1.f;
            sf::Color buildingColor = hsvToRgb(buildingHue, 0.7f, buildingVal);
            sf::Color buildingColorDark = hsvToRgb(buildingHue, 0.7f, buildingVal * 0.6f);
            
            // Front face
            sf::Vertex v;
            v.position = sf::Vector2f{mid.x - isoX, mid.y};
            v.color = buildingColor;
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x - isoX, mid.y};
            geom.buildings.append(v);
            
            // Top face
            v.color = buildingColorDark;
            v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - height - isoY};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - height - isoY};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x - isoX + isoY, mid.y - height - isoY};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x - isoX, mid.y - height};
            geom.buildings.append(v);
            
            // Side face
            v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - height - isoY};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - isoY};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - isoY};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y};
            geom.buildings.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y - height};
            geom.buildings.append(v);
            
            // Shadow
            v.color = sf::Color(0, 0, 0, 80);
            v.position = sf::Vector2f{mid.x - isoX + isoY, mid.y - isoY};
            geom.shadows.append(v);
            v.position = sf::Vector2f{mid.x + isoX + isoY, mid.y - isoY};
            geom.shadows.append(v);
            v.position = sf::Vector2f{mid.x + isoX, mid.y};
            geom.shadows.append(v);
            
            // Particles
            if (rng.next() > 0.7f) {
                float particleX = mid.x + (rng.next() - 0.5f) * 20.f;
                float particleY = mid.y - height + (rng.next() - 0.5f) * 10.f;
                float particleHue = seg.hue + rng.next() * 60.f;
                v.position = sf::Vector2f{particleX, particleY};
                v.color = hsvToRgb(particleHue, 0.8f, 0.9f);
                geom.particles.append(v);
            }
        }
        #endif
    }

    return geom;
}

// modes
enum class CityMode {
    Grid,
    Organic,
    Koch,
    Spiral,
    Tree
};

void configureLSystem(CityMode mode,
                      std::string& axiom,
                      std::unordered_map<char, std::string>& rules,
                      float& angleDeg,
                      float& stepLength)
{
    if (mode == CityMode::Grid) {
        // More blocky / grid-like
        axiom = "A";
        rules = {
            // Long avenues that spawn side streets
            {'A', "AF+BF-BF"},
            {'B', "F[+A]F[-A]"}
        };
        angleDeg = 90.f;
        stepLength = 12.f;
    } else if (mode == CityMode::Organic) {
        // More organic sprawl
        axiom = "A";
        rules = {
            {'A', "AFB"},
            {'B', "F[-B]F[+B]"}
        };
        angleDeg = 45.f;
        stepLength = 10.f;
    } else if (mode == CityMode::Spiral) {
        // Spiral city 
        //creates expanding spiral patterns
        axiom = "A";
        rules = {
            {'A', "F[++A][--A]F[+A]"},
            {'F', "FF"}
        };
        angleDeg = 20.f;  // Smaller angle for tighter spirals
        stepLength = 8.f;
    } else if (mode == CityMode::Tree) {
        // Tree-like structure 
        //more branching, less spiral
        axiom = "A";
        rules = {
            {'A', "F[+A][-A][++A][--A]"},
            {'F', "FF"}
        };
        angleDeg = 35.f;  // Larger angle for more spread-out branches
        stepLength = 10.f;
    } else { // Koch curve / snowflake
        axiom = "F--F--F";
        rules = {
            {'F', "F+F--F+F"}
        };
        angleDeg = 60.f;
        stepLength = 6.f;
    }
}


int main() {
    // Display OpenMP status at startup
    #ifdef _OPENMP
    std::cout << "========================================\n";
    std::cout << "  OpenMP PARALLELIZATION: ENABLED\n";
    std::cout << "  Threads available: " << omp_get_max_threads() << "\n";
    std::cout << "  Parallel sections: L-system generation, Building geometry\n";
    std::cout << "========================================\n\n";
    #else
    std::cout << "========================================\n";
    std::cout << "  OpenMP PARALLELIZATION: DISABLED\n";
    std::cout << "  Running in sequential mode\n";
    std::cout << "  (Compile with -fopenmp to enable parallelization)\n";
    std::cout << "========================================\n\n";
    #endif
    
    sf::RenderWindow window(sf::VideoMode({800u, 600u}), "Fractal City");
    window.setFramerateLimit(60);

    // Camera / view
    sf::View view = window.getDefaultView();

    // L-system / city parameters
    CityMode mode = CityMode::Grid;
    int iterations = 4; // fractal depth

    std::string axiom;
    std::unordered_map<char, std::string> rules;
    float angleDeg;
    float stepLength;

    configureLSystem(mode, axiom, rules, angleDeg, stepLength);

    //L-system string + growth state 
    std::string lsystemString;
    sf::Vector2f startPos{400.f, 300.f};

    std::size_t maxSymbols = 0;        // how many symbols we currently draw
    const float growSpeed = 250.f;     // symbols per second

    float baseHue = 0.f;               // animated over time
    const float hueScrollSpeed = 180.f; // how fast colors cycle
    const float hueStep = 2.0f;        // color difference between segments
    
    float totalTime = 0.f;             // total elapsed time for continuous motion
    float windStrength = 2.f;          // how much wind affects the structure

    sf::Clock clock; // for delta time

    auto rebuildString = [&]() {
        #ifdef _OPENMP
        std::cout << "[OpenMP] Generating L-system string (parallel)..." << std::flush;
        #endif
        lsystemString = generateLSystem(axiom, rules, iterations);
        #ifdef _OPENMP
        std::cout << " Done!\n";
        #endif
        
        const char* modeName = "Unknown";
        if (mode == CityMode::Grid) modeName = "Grid";
        else if (mode == CityMode::Organic) modeName = "Organic";
        else if (mode == CityMode::Koch) modeName = "Koch";
        else if (mode == CityMode::Spiral) modeName = "Spiral";
        else if (mode == CityMode::Tree) modeName = "Tree";
        std::cout << "Mode: " << modeName
                  << " | Iterations: " << iterations
                  << " | String length: " << lsystemString.size();
        #ifdef _OPENMP
        std::cout << " | Using " << omp_get_max_threads() << " threads";
        #endif
        std::cout << "\n";
        maxSymbols = 0; // restart growth
    };
    
    rebuildString();

    SceneGeometry geom{
        sf::VertexArray(sf::PrimitiveType::Lines),
        sf::VertexArray(sf::PrimitiveType::Lines)
    };

    while (window.isOpen()) {
        float dt = clock.restart().asSeconds();

        while (const std::optional<sf::Event> event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }

            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                // Quit
                if (key->code == sf::Keyboard::Key::Escape) {
                    window.close();
                }

                // Camera controls 
                const float panAmount = 30.f;

                if (key->code == sf::Keyboard::Key::W) {
                    view.move(sf::Vector2f{0.f, -panAmount});
                } else if (key->code == sf::Keyboard::Key::S) {
                    view.move(sf::Vector2f{0.f, panAmount});
                } else if (key->code == sf::Keyboard::Key::A) {
                    view.move(sf::Vector2f{-panAmount, 0.f});
                } else if (key->code == sf::Keyboard::Key::D) {
                    view.move(sf::Vector2f{panAmount, 0.f});
                } else if (key->code == sf::Keyboard::Key::Q) {
                    view.zoom(0.9f);  // zoom in
                } else if (key->code == sf::Keyboard::Key::E) {
                    view.zoom(1.1f);  // zoom out
                }

                //  Live angle tweak (morph shapes) 
                if (key->code == sf::Keyboard::Key::Left) {
                    angleDeg -= 5.f;
                } else if (key->code == sf::Keyboard::Key::Right) {
                    angleDeg += 5.f;
                }

                // Switch modes 
                if (key->code == sf::Keyboard::Key::Num1) {
                    mode = CityMode::Grid;
                    configureLSystem(mode, axiom, rules, angleDeg, stepLength);
                    rebuildString();
                } else if (key->code == sf::Keyboard::Key::Num2) {
                    mode = CityMode::Organic;
                    configureLSystem(mode, axiom, rules, angleDeg, stepLength);
                    rebuildString();
                } else if (key->code == sf::Keyboard::Key::Num3) {
                    mode = CityMode::Koch;
                    configureLSystem(mode, axiom, rules, angleDeg, stepLength);
                    rebuildString();
                } else if (key->code == sf::Keyboard::Key::Num4) {
                    mode = CityMode::Spiral;
                    configureLSystem(mode, axiom, rules, angleDeg, stepLength);
                    rebuildString();
                } else if (key->code == sf::Keyboard::Key::Num5) {
                    mode = CityMode::Tree;
                    configureLSystem(mode, axiom, rules, angleDeg, stepLength);
                    rebuildString();
                }
                
                // Wind control
                if (key->code == sf::Keyboard::Key::Up) {
                    windStrength += 0.5f;
                    if (windStrength > 10.f) windStrength = 10.f;
                } else if (key->code == sf::Keyboard::Key::Down) {
                    windStrength -= 0.5f;
                    if (windStrength < 0.f) windStrength = 0.f;
                }

                // Change iterations (fractal depth) 
                if (key->code == sf::Keyboard::Key::LBracket) { // '['
                    if (iterations > 1) {
                        iterations--;
                        rebuildString();
                    }
                } else if (key->code == sf::Keyboard::Key::RBracket) { // ']'
                    if (iterations < 10) { // reasonable upper bound
                        iterations++;
                        rebuildString();
                    }
                }
            }
        }

        // Update total time for continuous motion
        totalTime += dt;
        
        // animate growth (can loop for continuous effect)
        if (!lsystemString.empty() && maxSymbols < lsystemString.size()) {
            std::size_t increase = static_cast<std::size_t>(growSpeed * dt);
            if (increase == 0) increase = 1; // ensure progress
            maxSymbols += increase;
            if (maxSymbols > lsystemString.size())
                maxSymbols = lsystemString.size();
        }
        
        
        // animate color cycling
        baseHue += hueScrollSpeed * dt;
        
        // Make angle slowly rotate for continuous motion (subtle)
        angleDeg += std::sin(totalTime * 0.3f) * 0.1f * dt;

        // For Koch mode, no buildings; for city modes, add them
        bool addBuildings = (mode != CityMode::Koch);

        #ifdef _OPENMP
        // OpenMP will parallelize building generation inside buildGeometry
        #endif
        geom = buildGeometry(
            lsystemString,
            angleDeg,
            stepLength,
            startPos,
            maxSymbols,
            baseHue,
            hueStep,
            addBuildings,
            totalTime,
            windStrength
        );

        window.setView(view);

        window.clear(sf::Color(10, 10, 20)); // Slightly blue-black for depth
        window.draw(geom.shadows); // Draw shadows first
        window.draw(geom.roads);
        window.draw(geom.buildings);
        window.draw(geom.particles);
        window.display();
    }

    return 0;
}
