# Beam Internal Forces Analysis

A comprehensive structural analysis application built with Python and Pygame for visualizing and calculating internal forces in beams.

## Features

- **Interactive Beam Construction**: Click and drag to create beams of any length and orientation
- **Multiple Load Types**:
  - Point loads with adjustable magnitude and direction
  - Uniform line loads (distributed loads)
  - Trapezoidal/variable line loads
- **Support Systems**:
  - Fixed supports (prevents translation and rotation)
  - Roller supports (prevents translation in one direction)
  - Pin supports (prevents translation, allows rotation)
- **Real-time Analysis**:
  - Static determinacy checking
  - Support reaction calculations
  - Internal forces computation (Normal force N, Shear force Q, Moment M)
- **Visual Feedback**:
  - Animated force previews during load application
  - Internal force diagrams with filled areas
  - Support reaction arrows
  - Interactive scale slider for diagram scaling

## Requirements

- Python 3.7+
- pygame
- numpy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/sebastianschanz/beam-internal-forces.git
cd beam-internal-forces
```

2. Install required packages:
```bash
pip install pygame numpy
```

## Usage

Run the main application:
```bash
python beam-internal-forces-analysis.py
```

### Controls

- **B**: Beam creation mode
- **P**: Point load application mode
- **L**: Uniform line load mode
- **T**: Trapezoidal/variable line load mode
- **S**: Support placement/modification mode
- **C**: Clear all (reset)
- **D**: Toggle debug information
- **ESC**: Cancel current operation

### Workflow

1. **Create a Beam**: Press 'B' and click two points to define the beam
2. **Add Supports**: Press 'S' and click near beam ends to add supports (click multiple times to cycle through support types)
3. **Apply Loads**: Use 'P', 'L', or 'T' to add different types of loads
4. **Analyze**: The application automatically calculates and displays internal forces when the system is statically determinate
5. **Adjust View**: Use the scale slider (top-right) to adjust the diagram scaling

## Technical Details

### Static Analysis
The application implements classical structural analysis methods:
- Equilibrium equations (ΣFx=0, ΣFz=0, ΣM=0)
- Method of sections for internal forces
- Static determinacy verification (3n = s + v)

### Coordinate System
- Local beam coordinate system with x-axis along the beam
- z-axis perpendicular to the beam (for transverse forces)
- All calculations performed in local coordinates, then transformed for display

### Files
- `beam-internal-forces-analysis.py`: Main application with full functionality
- `static-base-game.py`: Extended version with additional features
- `connected-beams-demo.py`: Demo for connected beam structures
- `static-game.py`: Legacy version

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
