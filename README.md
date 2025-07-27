# Beam Analysis - Structural Engineering Tool

Interactive structural analysis software for beams with point loads, line loads, and various support conditions.

## 🚀 Quick Start

### **Option 1: Download & Run (Recommended)**
1. Download `BeamAnalysis.exe` from the [`dist/`](dist/) folder
2. Double-click to run - no installation required!
3. Compatible with Windows 7/8/10/11

### **Option 2: Run from Source**
```bash
pip install pygame numpy
python beam-analysis.py
```

## 🏗️ Features

- **Interactive Beam Design**: Click and drag to create beams
- **Loading Conditions**: 
  - Point loads (concentrated forces)
  - Line loads (distributed forces, uniform & variable)
- **Support Types**: Fixed, roller, and pin supports
- **Analysis Results**:
  - Support reaction calculations
  - Internal force diagrams (N, Q, M)
  - Static determinacy checking
- **Real-time Visualization**: Live preview and animated arrows

## 🎮 How to Use

1. **Create Beam**: Click and drag to draw a beam
2. **Add Supports**: Right-click on beam ends to cycle through support types
3. **Apply Loads**: 
   - Left-click for point loads
   - Hold Shift + drag for line loads
4. **Analysis**: Internal forces display automatically when statically determinate
5. **Delete**: Hold Ctrl and click on items to remove them

## 🔧 Controls

- **Mouse**: Primary interaction (click, drag, right-click)
- **Shift**: Hold for line load mode
- **Ctrl**: Hold for delete mode
- **Scroll**: Adjust force magnitudes and scale

## 📐 Engineering Accuracy

- Proper equilibrium calculations (ΣF = 0, ΣM = 0)
- Unit consistency (Forces in N, Moments in N⋅m)
- Support reaction calculations following structural mechanics principles
- Internal force diagrams with correct sign conventions

## 🛠️ Technical Details

- **Language**: Python with pygame and numpy
- **Size**: ~25-40 MB executable (includes all dependencies)
- **Performance**: Optimized with caching for smooth interaction
- **Requirements**: Windows only (for .exe), cross-platform for source

## 📦 File Structure

```
beam-analysis.py    # Main application source code
dist/
├── BeamAnalysis.exe # Ready-to-run executable
README.md          # This file
```

## 🎓 Educational Use

Perfect for:
- Structural engineering students
- Learning beam analysis concepts
- Quick structural calculations
- Teaching internal force diagrams
- Understanding support reactions

## 📜 License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- ✅ **Share** — copy and redistribute the material in any medium or format
- ✅ **Adapt** — remix, transform, and build upon the material

**Under the following terms:**
- 🏷️ **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- 🚫 **NonCommercial** — You may not use the material for commercial purposes

Perfect for educational use, research, and learning!

---

**Ready to analyze some beams?** Download `BeamAnalysis.exe` and start designing! 🏗️
