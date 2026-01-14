# Fractal City - L-System Visualization

An interactive fractal city generator using L-systems with optional OpenMP parallelization, built in C++ with SFML.

## Quick Start

This project is a **desktop SFML app**. 

- **Instructions page (GitHub Pages)**: open `demo.html` (or the Pages link)
- **Run the graphics app (local)**: compile `main.cpp`, then run `./fractal_city`

## Run Locally (macOS)

### 1) Install dependencies

- **SFML** (required): `brew install sfml`
- **OpenMP** (optional): `brew install libomp`

### 2) Compile

**With OpenMP (recommended if you installed libomp):**

```bash
g++ -std=c++17 -Xpreprocessor -fopenmp \
  -I/opt/homebrew/include -I/opt/homebrew/opt/libomp/include \
  -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
  -L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib -lomp \
  main.cpp -lsfml-graphics -lsfml-window -lsfml-system -o fractal_city
```

**Without OpenMP (works even if libomp is not installed):**

```bash
g++ -std=c++17 \
  -I/opt/homebrew/include \
  -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
  -L/opt/homebrew/lib \
  main.cpp -lsfml-graphics -lsfml-window -lsfml-system -o fractal_city
```

### 3) Run

```bash
./fractal_city
```

## Features

- **5 Different Modes**: Grid, Organic, Koch, Spiral, and Tree patterns
- **3D Isometric Buildings**: Buildings rendered with depth and shadows
- **OpenMP Parallelization**: Multi-threaded L-system generation and building geometry
- **Continuous Animation**: Wind effects and color cycling for dynamic visuals
- **Zoom + Camera**: Pan and zoom to explore the scene
- **Interactive Controls**: Real-time mode switching and parameter adjustment

## Controls

### Camera
- **W/A/S/D**: Pan camera
- **Q/E**: Zoom in/out
- **Left/Right Arrows**: Adjust angle (morph shapes)

### Modes
- **1**: Grid mode (90° angles, blocky city)
- **2**: Organic mode (45° angles, flowing sprawl)
- **3**: Koch mode (60° angles, snowflake pattern)
- **4**: Spiral mode (20° angles, spiral expansion)
- **5**: Tree mode (35° angles, branching structure)

### Parameters
- **[/]**: Decrease/increase iterations (fractal depth)
- **Up/Down Arrows**: Adjust wind strength
- **ESC**: Quit

## Technical Details

### Parallelization
- Uses OpenMP fork-join pattern
- Parallel L-system string generation (for large strings)
- Parallel building geometry generation
- Automatically uses all available CPU cores

### Requirements
- C++17 compiler
- SFML library
- OpenMP (optional, for parallelization)
- macOS/Linux (Windows requires different paths)

## How It Works

1. **L-System Generation**: Creates fractal strings using production rules
2. **Turtle Graphics**: Interprets the string to draw paths
3. **Geometry Building**: Generates roads and 3D buildings along paths
4. **Parallel Processing**: Uses OpenMP to speed up computation
5. **Rendering**: Displays with SFML graphics

## Creative Elements

- **3D Perspective**: Isometric buildings with shadows
- **Dynamic Colors**: HSV color cycling for visual appeal
- **Wind Effects**: Continuous motion through sine wave offsets
- **Particle Effects**: Floating elements around buildings
- **Organic Variation**: Random wobble and color variation

---

**Note**: The code automatically detects OpenMP and uses parallelization when available, falling back to sequential execution otherwise.
