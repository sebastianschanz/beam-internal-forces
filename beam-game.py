import pygame
import numpy as np
import sys
import math

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Beam Internal Forces Analysis")
clock = pygame.time.Clock()

# Color System - Clean and Organized
COLORS = {
    # Background and Grid
    'bg': (30, 30, 40),
    'grid': (60, 60, 70),
    
    # UI Elements
    'ui_text': (90, 150, 220),
    'ui_active': (255, 255, 255),
    'ui_bg': (80, 80, 90),
    'ui_highlight': (150, 230, 255),
    'status_ok': (90, 150, 220),
    'status_error': (255, 255, 60),
    
    # Structural Elements
    'beam': (170, 100, 190),
    'symbol_bg': (55, 55, 90),
    'symbol_line': (170, 100, 190),
    
    # Forces and Analysis
    'force_display': (210, 20, 80),
    'force_line': (210, 20, 80),
    'force_preview': (210, 20, 80),
    'force_text': (210, 20, 80),
    'reaction': (40, 40, 220),
    'reaction_text': (90, 150, 220),
    
    # Coordinate System
    'x_axis': (200, 40, 40),
    'z_axis': (30, 160, 30),
    
    # Internal Forces (N, Q, M)
    'N': (200, 150, 30),        # Normal force - red
    'Q': (100, 200, 100),       # Shear force - green  
    'M': (80, 140, 220),        # Moment - blue
    
    # Delete mode
    'delete_highlight': (255, 255, 60),
    'support_highlight': (180, 180, 30)
}

# Constants
GRID_SIZE = 25
FONTS = {
    'axis': ('consolas', 14),
    'values': ('consolas', 14),
    'reactions': ('consolas', 14),
    'debug': ('consolas', 12),
    'legend': ('consolas', 14),
    'slider': ('consolas', 14),
    'ui': ('consolas', 18),
    'preview': ('consolas', 28)
}

# Shared animation parameters for all oscillating arrows
ANIMATION_FREQUENCY = 2.0  # Oscillations per second
ANIMATION_AMPLITUDE = 3.0   # Pixels of oscillation amplitude (increased from 4.0)
ANIMATION_SEGMENTS = 50    # Segments for smooth curves (point loads use more)
ANIMATION_PERIOD_LENGTH = 50.0  # Pixels per complete sine wave cycle
MIN_ARROW_SPACING = GRID_SIZE   # Use grid distance (25px) for arrow spacing and threshold
LINE_LOAD_SPACING = GRID_SIZE * 2  # Use double grid distance (50px) for line load arrow spacing

# Force display parameters
ARROW_HEAD_SIZE = 14      # Length of arrow heads in pixels
ARROW_SIZE_RATIO = 0.3    # Arrow width ratio (narrower arrowheads)
FORCE_LINE_THICKNESS = 2  # Thickness of all force lines (slightly thicker)
ANIMATION_ARROW_HEAD_LENGTH = ARROW_HEAD_SIZE  # Use the new variable for animation arrow heads

# Performance optimization: Pre-cache fonts
_font_cache = {}
def get_font(font_key):
    """Get font by key with caching for performance"""
    if font_key not in _font_cache:
        family, size = FONTS[font_key]
        _font_cache[font_key] = pygame.font.SysFont(family, size)
    return _font_cache[font_key]

# Performance optimization: Geometry helper class
class GeometryCache:
    def __init__(self):
        self._arrow_geometry_cache = {}
        self._perpendicular_cache = {}
    
    def get_perpendicular_vector(self, vector):
        """Get perpendicular vector with caching"""
        key = (vector[0], vector[1])
        if key not in self._perpendicular_cache:
            self._perpendicular_cache[key] = np.array([-vector[1], vector[0]])
        return self._perpendicular_cache[key]
    
    def get_arrow_points(self, tip, direction_unit, arrow_length, arrow_width):
        """Generate arrow triangle points with caching"""
        key = (tip[0], tip[1], direction_unit[0], direction_unit[1], arrow_length, arrow_width)
        if key not in self._arrow_geometry_cache:
            perp = self.get_perpendicular_vector(direction_unit)
            base_center = tip - direction_unit * arrow_length
            left_base = base_center - perp * arrow_width
            right_base = base_center + perp * arrow_width
            self._arrow_geometry_cache[key] = [tip, left_base, right_base]
        return self._arrow_geometry_cache[key]
    
    def clear_cache(self):
        """Clear geometry cache when needed"""
        self._arrow_geometry_cache.clear()
        self._perpendicular_cache.clear()

# Global geometry cache instance
geometry_cache = GeometryCache()

def snap(pos):
    """Snap position to grid"""
    x, y = pos
    return np.array([
        round(x / GRID_SIZE) * GRID_SIZE,
        round(y / GRID_SIZE) * GRID_SIZE
    ], dtype=float)

def draw_transparent_polygon(surface, color, points, alpha=128):
    """Draw transparent polygon"""
    if len(points) < 3:
        return
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(temp_surface, (*color, alpha), points)
    surface.blit(temp_surface, (0, 0))

class Beam:
    def __init__(self, start, end):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.L = np.linalg.norm(self.end - self.start)  # Length using conventional L
        self.e_x = (self.end - self.start) / self.L  # Unit vector along beam
        self.e_z = np.array([-self.e_x[1], self.e_x[0]])  # Perpendicular unit vector
        self.point_loads = []  # Point loads: [(position, force_vector), ...]
        self.line_loads = []  # Line loads: [(start, end, end_amp, start_amp), ...]
        self.supports = {}  # Supports: {position: support_type}

    def global_to_local(self, v):
        return np.array([np.dot(v, self.e_x), np.dot(v, self.e_z)])

    def world_point(self, x):
        """Convert local x-coordinate to world coordinates"""
        return self.start + x * self.e_x

    def snap_to_beam(self, pos):
        """Snaps a point to the closest point on the beam"""
        # Vector from beam start to clicked point
        vec_to_point = pos - self.start
        # Projection onto the beam axis
        projection_length = np.dot(vec_to_point, self.e_x)
        # Limit to beam length
        projection_length = max(0, min(self.L, projection_length))
        # Calculate the point on the beam
        return self.start + projection_length * self.e_x

    def add_point_load(self, pos, direction):
        # Snap the point to the beam
        snapped_pos = self.snap_to_beam(pos)
        # Store the global coordinates directly
        self.point_loads.append((snapped_pos, direction))

    def add_line_load(self, start_pos, end_pos, end_amplitude, start_amplitude=None):
        # Snap both points to the beam
        snapped_start = self.snap_to_beam(start_pos)
        snapped_end = self.snap_to_beam(end_pos)
        
        # Line load is always perpendicular to the beam axis
        # Calculate the amplitudes in z-direction (perpendicular to beam)
        end_amplitude_z = np.dot(end_amplitude, self.e_z)
        
        if start_amplitude is not None:
            start_amplitude_z = np.dot(start_amplitude, self.e_z)
        else:
            # Uniform line load (old mode)
            start_amplitude_z = end_amplitude_z
        
        # Store start, end and both amplitudes
        self.line_loads.append((snapped_start, snapped_end, end_amplitude_z, start_amplitude_z))

    def add_support(self, pos):
        # Determine if it's at start or end
        dist_start = np.linalg.norm(pos - self.start)
        dist_end = np.linalg.norm(pos - self.end)
        
        if dist_start < dist_end:
            support_pos = "start"
            snap_pos = self.start
        else:
            support_pos = "end"
            snap_pos = self.end
            
        # Cycle through support types: 0->1->2->None->0...
        if support_pos in self.supports:
            current_type = self.supports[support_pos]
            if current_type == 2:  # Pin support -> no support
                del self.supports[support_pos]
            else:
                self.supports[support_pos] = current_type + 1
        else:
            self.supports[support_pos] = 0  # Fixed support
            
        return snap_pos

    def check_static_determinacy(self):
        """
        Checks static determinacy according to the formula: 3n = s + v
        n = Number of rigid bodies (here: 1 beam)
        s = Number of support reactions
        v = Number of joint reactions (here: 0, since only one beam)
        
        Return: (is_determinate, status_text)
        """
        n = 1  # One beam
        v = 0  # No joints
        s = 0  # Support reactions
        
        # Count support reactions
        for support_pos, support_type in self.supports.items():
            if support_type == 0:  # Fixed support
                s += 3  # Fx, Fz, M
            elif support_type == 1:  # Roller support
                s += 2  # Fx, Fz
            elif support_type == 2:  # Pin support
                s += 1  # Fz
        
        required_reactions = 3 * n  # = 3 for one beam
        
        if s < required_reactions:
            return False, f"Statically indeterminate: add support constraints ({s}<{required_reactions} DOF)"
        elif s > required_reactions:
            return False, f"Statically overdetermined: change support constraints ({s}>{required_reactions} DOF)"
        else:
            return True, f"Statically determinate ({s}={required_reactions} DOF)"

    def get_segments(self):
        """Creates segment division based on load positions and supports"""
        segments = [0, self.L]  # Start and end of beam
        
        # Add point loads
        for pos_global, _ in self.point_loads:
            x_l = np.dot(pos_global - self.start, self.e_x)
            segments.append(x_l)
        
        # Add line loads (start and end points)
        for start_pos, end_pos, end_amplitude, start_amplitude in self.line_loads:
            x1 = np.dot(start_pos - self.start, self.e_x)
            x2 = np.dot(end_pos - self.start, self.e_x)
            segments.extend([x1, x2])
        
        # Sort and remove duplicates
        segments = sorted(list(set(segments)))
        return segments

    def internal_forces(self, x):
        """
        Calculate internal forces at the negative side (right cut face)
        
        Physically correct calculation through equilibrium of the right beam part:
        - N(x) = -Σ Fx,i (all horizontal forces right of cut, with sign change)
        - Q(x) = -Σ Fz,i (all vertical forces right of cut, with sign change)  
        - M(x) = Σ Fz,i · (xi - x) + Σ M0,i (moments about cut location x from right part)
        
        This method is numerically more stable and physically clearer.
        
        Note: x is in pixels, but moments are calculated in proper units (N⋅m)
        """
        # Check static determinacy
        is_determinate, _ = self.check_static_determinacy()
        if not is_determinate:
            return 0, 0, 0
            
        # Get support reactions
        support_reactions = self.calculate_support_reactions()
        
        # Initialization
        N = Q = M = 0
        
        # 1. SUPPORT REACTIONS: All support forces right of the cut
        for support_pos, (fx, fz, m_support) in support_reactions.items():
            # Determine x-position of the support
            if support_pos == "start":
                x_l = 0
            elif support_pos == "end":
                x_l = self.L
            else:
                continue
            # Only consider supports right of the cut (exclusively x)
            if x_l > x:
                N += fx                          # Horizontal force (same direction)
                Q += fz                          # Vertical force (same direction)
                # Convert pixel distance to meters for moment calculation
                distance_m = (x_l - x) / GRID_SIZE  # Convert pixels to meters
                M -= fz * distance_m             # Force × lever arm in meters (negative for equilibrium)
                
                # Fixed support: add explicit moment
                if support_pos in self.supports and self.supports[support_pos] == 0:
                    M += m_support  # Reaction moment
        
        # 2. POINT LOADS: All point loads right of the cut
        for pos_global, force_global in self.point_loads:
            x_l = np.dot(pos_global - self.start, self.e_x)
            
            # Only loads right of the cut (exclusively x)
            if x_l > x:
                f_local = self.global_to_local(force_global)
                fx_point = f_local[0]           # Horizontal component
                fz_point = f_local[1]           # Vertical component
                
                N += fx_point                   # Normal force (same direction)
                Q += fz_point                   # Shear force (same direction)
                # Convert pixel distance to meters for moment calculation
                distance_m = (x_l - x) / GRID_SIZE  # Convert pixels to meters
                M -= fz_point * distance_m      # Moment = Force × lever arm in meters (negative for equilibrium)
        
        # 3. LINE LOADS: Calculate resultant force and moment right of the cut
        for start_pos, end_pos, end_amplitude, start_amplitude in self.line_loads:
            x1 = np.dot(start_pos - self.start, self.e_x)
            x2 = np.dot(end_pos - self.start, self.e_x)
            x1, x2 = min(x1, x2), max(x1, x2)  # Sortierung
            
            # Only consider active area to the right of the cut
            if x2 <= x:
                continue  # Complete load left of or at the cut
            
            x_left = max(x, x1)  # From cut or load start
            x_right = x2         # To load end
            
            if x_left < x_right:
                total_length = x2 - x1
                if total_length > 0:
                    # For linearly variable line load: q(ξ) = q1 + (q2-q1)*ξ/L
                    # where ξ is the local coordinate from 0 to L
                    q1 = start_amplitude / total_length  # Amplitude at start
                    q2 = end_amplitude / total_length    # Amplitude at end
                    
                    # Active length and local coordinates
                    l_active = x_right - x_left
                    xi1 = x_left - x1  # Local coordinate at cut
                    xi2 = x_right - x1 # Local coordinate at end
                    
                    # Resultant force of linear load in active region
                    # F = ∫[xi1 to xi2] (q1 + (q2-q1)*ξ/L) dξ
                    F_res = q1 * l_active + (q2-q1) * (xi2**2 - xi1**2) / (2 * total_length)
                    
                    # Center of gravity of linear load in active region
                    # x_s = (∫[xi1 to xi2] ξ*(q1 + (q2-q1)*ξ/L) dξ) / F_res
                    if abs(F_res) > 1e-12:
                        moment_integral = q1 * (xi2**2 - xi1**2) / 2 + (q2-q1) * (xi2**3 - xi1**3) / (3 * total_length)
                        x_centroid_local = moment_integral / F_res + x1
                    else:
                        x_centroid_local = (x_left + x_right) / 2
                    
                    # Contributions to internal forces
                    Q += F_res                                      # Resultant (same direction)
                    # Convert pixel distance to meters for moment calculation
                    distance_m = (x_centroid_local - x) / GRID_SIZE  # Convert pixels to meters
                    M -= F_res * distance_m         # Resultant × lever arm in meters (negative for equilibrium)
        
        # 4. BOUNDARY CONDITIONS: Moment at free ends and supports
        
        # At free end (without support) the moment must always be zero
        if abs(x - 0) < 1e-6 and "start" not in self.supports:
            M = 0  # Free end at start
        elif abs(x - self.L) < 1e-6 and "end" not in self.supports:
            M = 0  # Free end at end
            
        # At roller and pin supports: M = 0 (except fixed support)
        for support_pos, support_type in self.supports.items():
            if support_pos == "start":
                x_l = 0
            elif support_pos == "end":
                x_l = self.L
            else:
                continue
                
            # Moment at support position must be zero (except fixed support)
            if abs(x - x_l) < 1e-6 and support_type in [1, 2]:
                M = 0
        
        return N, Q, M
        
    def calculate_support_reactions(self):
        """
        Calculate support reactions systematically by setting up the 
        equilibrium equations: ΣFx=0, ΣFz=0, ΣM=0
        """
        # First check static determinacy
        is_determinate, _ = self.check_static_determinacy()
        if not is_determinate:
            return {}
            
        support_reactions = {}
        
        # Collect all external loads (without support reactions)
        sum_Fx = sum_Fz = sum_M_start = 0
        
        # 1. Point loads
        for pos_global, force_global in self.point_loads:
            f_local = self.global_to_local(force_global)
            x_l = np.dot(pos_global - self.start, self.e_x)
            
            sum_Fx += f_local[0]  # Horizontal force
            sum_Fz += f_local[1]  # Vertical force
            # Convert pixel distance to meters for moment calculation
            x_l_m = x_l / GRID_SIZE  # Convert pixels to meters
            sum_M_start += f_local[1] * x_l_m  # Moment about beam start in N⋅m
        
        # 2. Line loads
        for start_pos, end_pos, end_amplitude, start_amplitude in self.line_loads:
            x1 = np.dot(start_pos - self.start, self.e_x)
            x2 = np.dot(end_pos - self.start, self.e_x)
            x1, x2 = min(x1, x2), max(x1, x2)
            length = x2 - x1
            
            if length > 0:
                # For linearly variable line load
                # Resultant force = (q1 + q2) * L / 2
                q1 = start_amplitude / length  # Amplitude at start
                q2 = end_amplitude / length    # Amplitude at end
                
                sum_Fx += 0  # No horizontal component
                sum_Fz += (q1 + q2) * length / 2  # Vertical resultant
                
                # Moment of resultant about beam start
                # Center of gravity of linear load: x_s = x1 + L * (2*q2 + q1) / (3*(q1 + q2))
                if abs(q1 + q2) > 1e-12:
                    x_centroid = x1 + length * (2*q2 + q1) / (3*(q1 + q2))
                else:
                    x_centroid = x1 + length / 2
                # Convert pixel distance to meters for moment calculation
                x_centroid_m = x_centroid / GRID_SIZE  # Convert pixels to meters
                sum_M_start += (q1 + q2) * length / 2 * x_centroid_m  # Moment in N⋅m
        
        # 3. Solve equilibrium equations depending on support combination
        
        if len(self.supports) == 0:
            return support_reactions
            
        elif len(self.supports) == 1:
            # One support: Absorb all reactions
            support_pos = list(self.supports.keys())[0]
            support_type = self.supports[support_pos]
            
            if support_type == 0:  # Fixed support
                # Can absorb all reactions: Fx, Fz, M
                fx_support = -sum_Fx
                fz_support = -sum_Fz
                
                if support_pos == "start":
                    m_support = -sum_M_start
                else:  # End
                    # Calculate moment about endpoint
                    # Convert beam length to meters for moment calculation
                    beam_length_m = self.L / GRID_SIZE  # Convert pixels to meters
                    m_about_end = sum_M_start - sum_Fz * beam_length_m
                    m_support = -m_about_end
                    
                support_reactions[support_pos] = (fx_support, fz_support, m_support)
                
            elif support_type == 1:  # Roller support
                # Can absorb Fx and Fz, but no moment
                support_reactions[support_pos] = (-sum_Fx, -sum_Fz, 0)
                
            elif support_type == 2:  # Pin support
                # Can only absorb Fz
                support_reactions[support_pos] = (0, -sum_Fz, 0)
                
        elif len(self.supports) == 2:
            # Two supports: Systematic solution of equilibrium equations
            support_positions = list(self.supports.keys())
            
            if "start" in support_positions and "end" in support_positions:
                type_start = self.supports["start"]
                type_end = self.supports["end"]
                
                # Standard case: Roller/Pin supports at both ends
                if type_start in [1, 2] and type_end in [1, 2]:
                    
                    # Moment equilibrium about start: All moments about start: loads + end reaction × beam length = 0
                    # fz_end × L + sum_M_start = 0
                    # Convert beam length to meters for moment calculation
                    beam_length_m = self.L / GRID_SIZE  # Convert pixels to meters
                    fz_end = -sum_M_start / beam_length_m if beam_length_m > 0 else 0
                    
                    # Force equilibrium vertically: ΣFz = 0
                    # fz_start + fz_end + sum_Fz = 0
                    fz_start = -(sum_Fz + fz_end)
                    
                    # Force equilibrium horizontally: ΣFx = 0
                    # Pin support cannot take horizontal force
                    if type_start == 2:  # Start is pin support
                        fx_start = 0
                        fx_end = -sum_Fx
                    elif type_end == 2:  # End is pin support
                        fx_start = -sum_Fx
                        fx_end = 0
                    else:  # Both roller supports - distribution possible
                        fx_start = -sum_Fx / 2
                        fx_end = -sum_Fx / 2
                    
                    support_reactions["start"] = (fx_start, fz_start, 0)
                    support_reactions["end"] = (fx_end, fz_end, 0)
                    
                # Special case: One fixed support
                elif type_start == 0 or type_end == 0:
                    if type_start == 0:  # Fixed support at start
                        fx_start = -sum_Fx
                        fz_start = -sum_Fz
                        # Use moment about start point directly
                        m_start = -sum_M_start
                        support_reactions["start"] = (fx_start, fz_start, m_start)
                        
                        # Second support redundant/overdetermined
                        support_reactions["end"] = (0, 0, 0)
                    else:  # Fixed support at end
                        fx_end = -sum_Fx
                        fz_end = -sum_Fz
                        # Moment about endpoint: All loads create moment about end
                        m_about_end = 0
                        
                        # Point loads: Moment about end
                        for pos_global, force_global in self.point_loads:
                            f_local = self.global_to_local(force_global)
                            x_l = np.dot(pos_global - self.start, self.e_x)
                            # Convert pixel distance to meters for moment calculation
                            distance_from_end_m = (self.L - x_l) / GRID_SIZE  # Convert pixels to meters
                            m_about_end += f_local[1] * distance_from_end_m  # Distance from end in meters
                        
                        # Line loads: Moment about end
                        for start_pos, end_pos, amplitude in self.line_loads:
                            x1 = np.dot(start_pos - self.start, self.e_x)
                            x2 = np.dot(end_pos - self.start, self.e_x)
                            x1, x2 = min(x1, x2), max(x1, x2)
                            length = x2 - x1
                            
                            if length > 0:
                                q = amplitude / length
                                x_centroid = x1 + length / 2
                                F_res = q * length
                                # Convert pixel distance to meters for moment calculation
                                distance_from_end_m = (self.L - x_centroid) / GRID_SIZE  # Convert pixels to meters
                                m_about_end += F_res * distance_from_end_m  # Distance from end in meters
                        
                        m_end = -m_about_end  # Reaction moment
                        support_reactions["end"] = (fx_end, fz_end, m_end)
                        
                        # First support redundant/overdetermined
                        support_reactions["start"] = (0, 0, 0)
        
        return support_reactions

    def draw_support(self, surf, pos, support_type, is_highlighted=False):
        """Draws a support at the given position with official support symbols in line-graphics style"""
        
        # Determine colors based on highlighting
        bg_color = COLORS['delete_highlight'] if is_highlighted else COLORS['symbol_bg']
        symbol_line_color = COLORS['support_highlight'] if is_highlighted else COLORS['symbol_line']
        
        # Offset for better centering (since hatching goes down-right)
        offset = np.array([-2, -2])  # Shift slightly up-left
        centered_pos = pos + offset
        
        if support_type == 0:  # Fixed support - Hatch with vertical line
            # Background circle (highlighted if selected)
            pygame.draw.circle(surf, bg_color, pos.astype(int), 20)
            
            # Main beam (always vertical, independent of beam direction)
            start_pos = centered_pos + np.array([0, -10])  # 10 pixels up
            end_pos = centered_pos + np.array([0, 10])     # 10 pixels down
            pygame.draw.line(surf, symbol_line_color, start_pos.astype(int), end_pos.astype(int), 2)
            
            # Standardized hatching (always down-right)
            base_y = centered_pos[1] + 10  # Below the beam
            for i in range(5):  # Fewer lines for smaller symbols
                x_pos = centered_pos[0] - 6 + i * 3
                pygame.draw.line(surf, symbol_line_color, (x_pos, base_y), (x_pos + 2, base_y + 5), 2)
                
        elif support_type == 1:  # Roller support - Triangle with circle on top, hatching below
            # Background circle (highlighted if selected)
            pygame.draw.circle(surf, bg_color, pos.astype(int), 20)
            
            # Slight shift right for better balance
            symbol_offset = np.array([1, 0])
            symbol_pos = centered_pos + symbol_offset
            
            # Triangle (outline only, line-graphics style) - always same orientation
            triangle_points = [
                (symbol_pos[0], symbol_pos[1] - 8),      # Top point
                (symbol_pos[0] - 9, symbol_pos[1] + 6),  # Bottom left, smaller
                (symbol_pos[0] + 9, symbol_pos[1] + 6)   # Bottom right, smaller
            ]
            pygame.draw.polygon(surf, symbol_line_color, triangle_points, 2)  # Use highlight color when highlighted
            
            # Circle at the tip (line-graphics) - smaller, slightly offset right and slightly larger
            circle_offset = 1  # Additional offset right for the circle
            pygame.draw.circle(surf, symbol_line_color, (int(symbol_pos[0] + circle_offset), int(symbol_pos[1] - 8)), 3, 2)
            
            # Hatching at the base (clean parallel lines) - always down-right
            base_y = symbol_pos[1] + 6
            for i in range(5):  # Fewer lines for smaller symbols
                x_pos = symbol_pos[0] - 6 + i * 3
                pygame.draw.line(surf, symbol_line_color, (x_pos, base_y), (x_pos + 2, base_y + 5), 2)
                
        elif support_type == 2:  # Pin support - Triangle with circle, offset hatching
            # Background circle (highlighted if selected)
            pygame.draw.circle(surf, bg_color, pos.astype(int), 20)
            
            # Slight shift right for better balance
            symbol_offset = np.array([1, 0])
            symbol_pos = centered_pos + symbol_offset
            
            # Triangle (outline only, line-graphics style) - always same orientation
            triangle_points = [
                (symbol_pos[0], symbol_pos[1] - 8),      # Top point
                (symbol_pos[0] - 9, symbol_pos[1] + 6),  # Bottom left, smaller
                (symbol_pos[0] + 9, symbol_pos[1] + 6)   # Bottom right, smaller
            ]
            pygame.draw.polygon(surf, symbol_line_color, triangle_points, 2)  # Use highlight color when highlighted
            
            # Circle at the tip (line-graphics) - smaller, slightly offset right and slightly larger
            circle_offset = 1  # Additional offset right for the circle
            pygame.draw.circle(surf, symbol_line_color, (int(symbol_pos[0] + circle_offset), int(symbol_pos[1] - 8)), 3, 2)
            
            # Offset hatching (with clear gap for mobility) - always down-right
            base_y = symbol_pos[1] + 11  # Further down for visible gap
            for i in range(5):  # Fewer lines for smaller symbols
                x_pos = symbol_pos[0] - 6 + i * 3
                pygame.draw.line(surf, symbol_line_color, (x_pos, base_y), (x_pos + 2, base_y + 5), 2)
            


    def draw(self, surf, highlight_item=None):
        # Determine colors based on highlighting
        beam_color = COLORS['delete_highlight'] if highlight_item == ('beam', 0) else COLORS['beam']
        
        # Draw beam as rectangle for perpendicular ends
        thickness = 20  # Noch dicker (war 16)
        
        # Calculate the four corner points of the rectangle
        half_thickness = thickness / 2
        offset = self.e_z * half_thickness
        
        corners = [
            self.start + offset,
            self.start - offset,
            self.end - offset,
            self.end + offset
        ]
        
        pygame.draw.polygon(surf, beam_color, corners)  # Use highlight color if beam is highlighted
        
        # Draw coordinate system at starting point
        x_axis_end = self.start + self.e_x * 40
        z_axis_end = self.start + self.e_z * 40
        
        # x-Achse (rot)
        pygame.draw.line(surf, COLORS['x_axis'], self.start, x_axis_end, 2)
        
        # z-Achse (blau) 
        pygame.draw.line(surf, COLORS['z_axis'], self.start, z_axis_end, 2)
        
        # Achsenbeschriftung
        font_axis = get_font('axis')
        x_text = font_axis.render("x", True, COLORS['x_axis'])
        z_text = font_axis.render("z", True, COLORS['z_axis'])
        surf.blit(x_text, (x_axis_end + np.array([5, -10])).astype(int))
        surf.blit(z_text, (z_axis_end + np.array([5, -10])).astype(int))
        
        # Draw point loads with values
        for i, (pos_global, force_global) in enumerate(self.point_loads):
            # Determine colors based on highlighting
            is_highlighted = highlight_item == ('point_load', i)
            line_color = COLORS['delete_highlight'] if is_highlighted else COLORS['force_line']
            
            # Draw arrow
            tip = pos_global + force_global
            pygame.draw.line(surf, line_color, pos_global, tip, FORCE_LINE_THICKNESS)
            
            # Optimized arrow head drawing
            force_norm = np.linalg.norm(force_global)
            if force_norm > 0:
                force_unit = force_global / force_norm
                triangle_width = ARROW_HEAD_SIZE * ARROW_SIZE_RATIO
                
                # Use geometry cache for arrow points
                arrow_points = geometry_cache.get_arrow_points(
                    tip, force_unit, ARROW_HEAD_SIZE, triangle_width
                )
                pygame.draw.polygon(surf, line_color, arrow_points)
            
            # Display force value (proportional to arrow length) - with distance in force direction
            force_value = force_norm
            font_values = get_font('values')
            # Use highlight color for text when point load is highlighted
            text_color = COLORS['delete_highlight'] if is_highlighted else COLORS['force_text']
            text = font_values.render(f"{force_value:.0f}N", True, text_color)
            
            # Position text in force direction with distance from arrow tip
            if force_norm > 0:
                force_unit = force_global / force_norm
                text_offset = force_unit * 15  # 15 pixel distance in force direction
                text_pos = tip + text_offset
            else:
                text_pos = tip + np.array([5, -15])
                
            # Center the text at the position like preview does
            text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            surf.blit(text, text_rect)
            
        # Draw line loads with values
        for i, (start_pos, end_pos, end_amplitude, start_amplitude) in enumerate(self.line_loads):
            # Determine colors based on highlighting
            is_highlighted = highlight_item == ('line_load', i)
            line_color = COLORS['delete_highlight'] if is_highlighted else COLORS['force_line']
            
            # Pre-calculate common values for performance
            force_vector_end = self.e_z * end_amplitude
            force_vector_start = self.e_z * start_amplitude
            length = np.linalg.norm(end_pos - start_pos)
            
            # Draw the trapezoid/triangle for the variable line load
            if abs(start_amplitude) > 1e-6 or abs(end_amplitude) > 1e-6:
                rect_points = [start_pos, end_pos, end_pos + force_vector_end, start_pos + force_vector_start]
                pygame.draw.polygon(surf, line_color, rect_points, FORCE_LINE_THICKNESS)
                draw_transparent_polygon(surf, line_color, rect_points, 50)
            
            # Pre-calculate arrow distribution
            if length < LINE_LOAD_SPACING * 2:
                num_arrows = 2
            else:
                num_arrows = max(2, int(length / LINE_LOAD_SPACING) + 1)
            
            # Pre-calculate step and amplitude difference for interpolation
            if num_arrows > 1:
                step = 1.0 / (num_arrows - 1)
                amplitude_diff = end_amplitude - start_amplitude
            else:
                step = 0
                amplitude_diff = 0
            
            # Draw arrows with optimized calculations
            for j in range(num_arrows):
                t = j * step if num_arrows > 1 else 0
                arrow_start = start_pos + t * (end_pos - start_pos)
                
                # Linear interpolation of amplitude
                current_amplitude = start_amplitude + t * amplitude_diff
                force_vector = self.e_z * current_amplitude
                arrow_end = arrow_start + force_vector
                
                # Draw arrow if significant
                force_norm = np.linalg.norm(force_vector)
                if force_norm > 1e-6:
                    pygame.draw.line(surf, line_color, arrow_start, arrow_end, FORCE_LINE_THICKNESS)
                    
                    # Optimized arrow head using geometry cache
                    force_unit = force_vector / force_norm
                    triangle_width = ARROW_HEAD_SIZE * ARROW_SIZE_RATIO
                    arrow_points = geometry_cache.get_arrow_points(
                        arrow_end, force_unit, ARROW_HEAD_SIZE, triangle_width
                    )
                    pygame.draw.polygon(surf, line_color, arrow_points)
            
            # Display force value - show start and end values for variable load
            font_values = get_font('values')
            # Use highlight color for text when line load is highlighted
            text_color = COLORS['delete_highlight'] if is_highlighted else COLORS['force_text']
            
            if abs(start_amplitude - end_amplitude) < 1e-6:
                # Uniform load
                force_per_meter = abs(end_amplitude) / (length / 1000) if length > 0 else 0
                text = font_values.render(f"{force_per_meter:.0f}N/m", True, text_color)
                
                # Position text in the middle
                mid_pos = (start_pos + end_pos) / 2
                if abs(end_amplitude) > 0:
                    force_unit = force_vector_end / np.linalg.norm(force_vector_end)
                    text_offset = force_unit * 15
                    text_pos = mid_pos + force_vector_end + text_offset
                else:
                    text_pos = mid_pos + np.array([5, -5])
                # Center the text at the position like preview does
                text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text, text_rect)
            else:
                # Variable load - always show both values (including 0 N/m)
                start_force_per_meter = abs(start_amplitude) / (length / 1000) if length > 0 else 0
                end_force_per_meter = abs(end_amplitude) / (length / 1000) if length > 0 else 0
                
                # Start value - always show, including 0 N/m
                text_start = font_values.render(f"{start_force_per_meter:.0f}N/m", True, text_color)
                if abs(start_amplitude) > 0:
                    force_unit = force_vector_start / np.linalg.norm(force_vector_start)
                    text_offset = force_unit * 15
                    text_pos = start_pos + force_vector_start + text_offset
                else:
                    text_pos = start_pos + np.array([5, -15])  # Position for zero values
                # Center the text at the position like preview does
                text_rect = text_start.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text_start, text_rect)
                
                # End value - always show, including 0 N/m
                text_end = font_values.render(f"{end_force_per_meter:.0f}N/m", True, text_color)
                if abs(end_amplitude) > 0:
                    force_unit = force_vector_end / np.linalg.norm(force_vector_end)
                    text_offset = force_unit * 15
                    text_pos = end_pos + force_vector_end + text_offset
                else:
                    text_pos = end_pos + np.array([5, -15])  # Position for zero values
                # Center the text at the position like preview does
                text_rect = text_end.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text_end, text_rect)
            
        # Draw support reactions as arrows (only for static determinacy)
        is_determinate, _ = self.check_static_determinacy()
        if is_determinate:
            support_reactions = self.calculate_support_reactions()
            for support_pos, (fx, fz, m_support) in support_reactions.items():
                pos = self.start if support_pos == "start" else self.end
                
                # Reaction force in z-direction (optimized)
                if abs(fz) > 0.1:
                    force_vector = self.e_z * fz * 0.5
                    tip = pos + force_vector
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, FORCE_LINE_THICKNESS)
                    
                    # Optimized arrow head for z-direction
                    force_unit = force_vector / np.linalg.norm(force_vector)
                    triangle_width = 6 * ARROW_SIZE_RATIO  # Smaller for reaction forces
                    arrow_points = geometry_cache.get_arrow_points(tip, force_unit, 6, triangle_width)
                    pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)
                    
                    # Text rendering
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fz:.0f}N", True, COLORS['reaction'])
                    surf.blit(text, (tip + np.array([5, -10])).astype(int))
                    
                # Reaction force in x-direction (optimized)
                if abs(fx) > 0.1:
                    force_vector = self.e_x * fx * 0.5
                    tip = pos + force_vector
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, FORCE_LINE_THICKNESS)
                    
                    # Optimized arrow head for x-direction
                    force_unit = force_vector / np.linalg.norm(force_vector)
                    triangle_width = 6 * ARROW_SIZE_RATIO
                    arrow_points = geometry_cache.get_arrow_points(tip, force_unit, 6, triangle_width)
                    pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)
                    
                    # Text rendering
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fx:.0f}N", True, COLORS['reaction'])
                    surf.blit(text, (tip + np.array([5, 5])).astype(int))
                    
        # Draw supports last so they overlay everything (including graphs)
        for support_pos, support_type in self.supports.items():
            is_highlighted = highlight_item == ('support', support_pos)
            if support_pos == "start":
                self.draw_support(surf, self.start, support_type, is_highlighted)
            elif support_pos == "end":
                self.draw_support(surf, self.end, support_type, is_highlighted)

    def debug_internal_forces(self, surf):
        """
        Debugging function for internal forces calculation
        Shows numerical values at critical locations
        """
        if not hasattr(self, 'supports') or len(self.supports) == 0:
            return
            
        is_determinate, _ = self.check_static_determinacy()
        if not is_determinate:
            return
            
        # Show debugging info - top left under shortcuts, aligned with shortcuts
        font_debug = get_font('debug')
        debug_y = 60  # Under the second shortcut line (10 + 25 + 25 pixel spacing)
        debug_x = 10  # Left-aligned with shortcuts
        
        # Show support reactions
        support_reactions = self.calculate_support_reactions()
        debug_text = font_debug.render("=== SUPPORT REACTIONS ===", True, COLORS['force_text'])
        surf.blit(debug_text, (debug_x, debug_y))
        debug_y += 20
        
        for support_pos, (fx, fz, m) in support_reactions.items():
            text = f"{support_pos}: Fx={fx:.1f}N, Fz={fz:.1f}N, M={m:.1f}Nm"
            debug_text = font_debug.render(text, True, COLORS['force_text'])
            surf.blit(debug_text, (debug_x, debug_y))
            debug_y += 15
            
        debug_y += 10
        debug_text = font_debug.render("=== INTERNAL FORCES ===", True, COLORS['force_text'])
        surf.blit(debug_text, (debug_x, debug_y))
        debug_y += 20
        
        # Internal forces at critical points
        test_points = [0, self.L/4, self.L/2, 3*self.L/4, self.L]
        for x in test_points:
            N, Q, M = self.internal_forces(x)
            text = f"x={x:.1f}: N={N:.1f}N, Q={Q:.1f}N, M={M:.1f}Nm"
            debug_text = font_debug.render(text, True, COLORS['force_text'])
            surf.blit(debug_text, (debug_x, debug_y))
            debug_y += 15

    def draw_diagrams(self, surf, scale_factor=0.01):
        if self.L == 0:
            return
            
        # Check static determinacy
        is_determinate, status_text = self.check_static_determinacy()
        
        if not is_determinate:
            # No warning displayed here - already shown top left
            return
        
        # Segment division for accurate internal force diagrams    
        segments = self.get_segments()
        
        pts_N, pts_Q, pts_M = [], [], []
        beam_line_points = []
        
        # Generate enough points for each segment
        for i in range(len(segments) - 1):
            x_start = segments[i]
            x_end = segments[i + 1]
            
            # At least 5 points per segment, more for longer segments
            num_points = max(5, int((x_end - x_start) / self.L * 100))
            
            for j in range(num_points + 1):
                if j == num_points and i < len(segments) - 2:
                    continue  # Skip last point except for last segment
                    
                t = j / num_points if num_points > 0 else 0
                x = x_start + t * (x_end - x_start)
                w = self.world_point(x)
                N, Q, M = self.internal_forces(x)
                
                # Beam centerline points for polygon filling
                beam_line_points.append(w)
                
                # Scale internal forces simply with scale_factor
                pts_N.append(w + self.e_z * N * scale_factor)
                pts_Q.append(w + self.e_z * Q * scale_factor)
                # Use smaller scaling for moment display (visual only, actual values remain correct)
                pts_M.append(w + self.e_z * M * scale_factor * 0.1)
        
        # Check if graphs have non-zero values (check distance from beam line, not just Y-coordinate)
        has_N_values = any(np.linalg.norm(N_pt - beam_pt) > 0.1 for N_pt, beam_pt in zip(pts_N, beam_line_points))
        has_Q_values = any(np.linalg.norm(Q_pt - beam_pt) > 0.1 for Q_pt, beam_pt in zip(pts_Q, beam_line_points))
        has_M_values = any(np.linalg.norm(M_pt - beam_pt) > 0.1 for M_pt, beam_pt in zip(pts_M, beam_line_points))
        
        # Draw areas under curves with 50% transparency (only if not zero)
        if len(pts_N) > 1 and len(beam_line_points) > 1 and has_N_values:
            # N-area (red with 50% transparency)
            n_polygon = pts_N + list(reversed(beam_line_points))
            draw_transparent_polygon(surf, COLORS['N'], n_polygon, 70)
            
        if len(pts_Q) > 1 and len(beam_line_points) > 1 and has_Q_values:
            # Q-area (green with 50% transparency)
            q_polygon = pts_Q + list(reversed(beam_line_points))
            draw_transparent_polygon(surf, COLORS['Q'], q_polygon, 70)
            
        if len(pts_M) > 1 and len(beam_line_points) > 1 and has_M_values:
            # M-area (blue with 50% transparency)
            m_polygon = pts_M + list(reversed(beam_line_points))
            draw_transparent_polygon(surf, COLORS['M'], m_polygon, 70)
        
        # Draw internal force diagrams (lines over areas, only if not zero)
        if len(pts_N) > 1 and has_N_values:
            pygame.draw.lines(surf, COLORS['N'], False, pts_N, 2)
        if len(pts_Q) > 1 and has_Q_values:
            pygame.draw.lines(surf, COLORS['Q'], False, pts_Q, 2)
        if len(pts_M) > 1 and has_M_values:
            pygame.draw.lines(surf, COLORS['M'], False, pts_M, 2)
            
        # Dynamic labeling directly on the graphs
        font_legend = get_font('legend')
        
        # Labels only if statically determinate and graphs present and not zero
        if is_determinate:
            # N(x) label - right-aligned at the height of the first graph point
            if len(pts_N) > 5 and has_N_values:
                n_pos = pts_N[0]  # First point of the graph
                n_text = font_legend.render("N(x)", True, COLORS['N'])
                # Right-align so larger text extends to the left
                text_rect = n_text.get_rect()
                text_rect.right = int(n_pos[0] - 5)  # 5 pixels to the left of graph point
                text_rect.centery = int(n_pos[1])    # Exactly at graph point height
                surf.blit(n_text, text_rect)
            
            # Q(x) label - right-aligned at the height of the first graph point
            if len(pts_Q) > 5 and has_Q_values:
                q_pos = pts_Q[0]  # First point of the graph
                q_text = font_legend.render("Q(x)", True, COLORS['Q'])
                # Right-align so larger text extends to the left
                text_rect = q_text.get_rect()
                text_rect.right = int(q_pos[0] - 5)  # 5 pixels to the left of graph point
                text_rect.centery = int(q_pos[1])    # Exactly at graph point height
                surf.blit(q_text, text_rect)
            
            # M(x) label - right-aligned at the height of the first graph point
            if len(pts_M) > 5 and has_M_values:
                m_pos = pts_M[0]  # First point of the graph
                m_text = font_legend.render("M(x)", True, COLORS['M'])
                # Right-align so larger text extends to the left
                text_rect = m_text.get_rect()
                text_rect.right = int(m_pos[0] - 5)  # 5 pixels to the left of graph point
                text_rect.centery = int(m_pos[1])    # Exactly at graph point height
                surf.blit(m_text, text_rect)
        
        # Add significant values to the graphs
        self.draw_significant_values(surf, segments, scale_factor)

    def filter_significant_points(self, significant_points):
        """Filter significant points to avoid visual clutter"""
        if not significant_points:
            return []
        
        # Minimum distance between points (in pixels along the beam)
        min_distance = 40  # Pixels - adjust as needed
        
        # Group points by force type
        points_by_type = {'N': [], 'Q': [], 'M': []}
        for point in significant_points:
            x, value, force_type, point_type = point
            points_by_type[force_type].append(point)
        
        filtered_points = []
        
        # Process each force type separately
        for force_type in ['N', 'Q', 'M']:
            type_points = points_by_type[force_type]
            if not type_points:
                continue
                
            # Sort points by x position
            type_points.sort(key=lambda p: p[0])
            
            # Priority order: zero_point > zero > extremum > start/end
            priority_order = {'zero_point': 0, 'zero': 1, 'extremum': 2, 'start': 3, 'end': 3}
            
            # Filter points by minimum distance and priority
            last_x = -float('inf')
            for point in type_points:
                x, value, force_type, point_type = point
                
                # Always keep zero crossings and zero points (highest priority)
                if point_type in ['zero', 'zero_point']:
                    filtered_points.append(point)
                    last_x = x
                # For other points, check distance and significance
                elif x - last_x >= min_distance:
                    # Only keep significant extrema (not tiny values)
                    if point_type == 'extremum' and abs(value) > 1.0:  # At least 1N or 1Nm
                        filtered_points.append(point)
                        last_x = x
                    # Keep start/end points if they're significant
                    elif point_type in ['start', 'end'] and abs(value) > 0.5:  # At least 0.5N or 0.5Nm
                        filtered_points.append(point)
                        last_x = x
        
        # Limit total number of points per force type to avoid overwhelming display
        max_points_per_type = 5
        final_points = []
        
        # Group filtered points by type again and limit count
        filtered_by_type = {'N': [], 'Q': [], 'M': []}
        for point in filtered_points:
            force_type = point[2]
            filtered_by_type[force_type].append(point)
        
        for force_type in ['N', 'Q', 'M']:
            type_points = filtered_by_type[force_type]
            if len(type_points) <= max_points_per_type:
                final_points.extend(type_points)
            else:
                # If too many points, prioritize by importance
                type_points.sort(key=lambda p: (priority_order.get(p[3], 4), -abs(p[1])))
                final_points.extend(type_points[:max_points_per_type])
        
        return final_points

    def draw_significant_values(self, surf, segments, scale_factor):
        """Draw significant values (max, min, zero crossings) on internal force diagrams"""
        if not segments:
            return
            
        font_values = get_font('values')  # Use same font as force values
        
        # New robust approach: analyze each segment separately
        significant_points = []
        
        # Process each segment between discontinuities
        for i in range(len(segments) - 1):
            x_start = segments[i]
            x_end = segments[i + 1]
            segment_length = x_end - x_start
            
            if segment_length < 1e-6:  # Skip tiny segments
                continue
            
            # Sample the segment densely to understand its behavior
            num_samples = max(10, int(segment_length / 10))  # At least 10 samples, more for longer segments
            sample_x = []
            sample_N = []
            sample_Q = []
            sample_M = []
            
            for j in range(num_samples + 1):
                t = j / num_samples if num_samples > 0 else 0
                x = x_start + t * segment_length
                N, Q, M = self.internal_forces(x)
                sample_x.append(x)
                sample_N.append(N)
                sample_Q.append(Q)
                sample_M.append(M)
            
            # Analyze each force type in this segment
            for force_type, values in [('N', sample_N), ('Q', sample_Q), ('M', sample_M)]:
                if not values:
                    continue
                
                # Check if segment has significant variation
                max_val = max(values)
                min_val = min(values)
                variation = abs(max_val - min_val)
                
                # Skip if all values are essentially zero
                if abs(max_val) < 0.01 and abs(min_val) < 0.01:
                    continue
                
                # For segments with little variation (constant or nearly constant)
                if variation < max(0.1, abs(max_val) * 0.05):  # Less than 5% variation or 0.1 units
                    # Only show one value per constant segment (at midpoint)
                    mid_idx = len(values) // 2
                    mid_x = sample_x[mid_idx]
                    mid_val = values[mid_idx]
                    if abs(mid_val) > 0.01:  # Only if significant
                        significant_points.append((mid_x, mid_val, force_type, 'constant'))
                else:
                    # For segments with variation, find extrema
                    # Find absolute maximum
                    max_idx = values.index(max_val)
                    if abs(max_val) > 0.01:
                        significant_points.append((sample_x[max_idx], max_val, force_type, 'maximum'))
                    
                    # Find absolute minimum (only if different from maximum)
                    min_idx = values.index(min_val)
                    if abs(min_val) > 0.01 and min_idx != max_idx:
                        significant_points.append((sample_x[min_idx], min_val, force_type, 'minimum'))
        
        # Add discontinuity points (segment boundaries where values change significantly)
        for i in range(1, len(segments) - 1):  # Skip first and last
            x_boundary = segments[i]
            
            # Check values just before and after the boundary
            x_before = x_boundary - 1e-3
            x_after = x_boundary + 1e-3
            
            # Make sure we're within beam bounds
            if x_before < 0:
                x_before = 0
            if x_after > self.L:
                x_after = self.L
            
            N_before, Q_before, M_before = self.internal_forces(x_before)
            N_after, Q_after, M_after = self.internal_forces(x_after)
            
            # Check for significant jumps in each force type
            for force_type, val_before, val_after in [('N', N_before, N_after), 
                                                     ('Q', Q_before, Q_after), 
                                                     ('M', M_before, M_after)]:
                jump = abs(val_after - val_before)
                if jump > max(0.1, abs(max(val_before, val_after)) * 0.1):  # Significant jump
                    # Show the larger magnitude value
                    if abs(val_after) > abs(val_before):
                        significant_points.append((x_boundary, val_after, force_type, 'discontinuity'))
                    elif abs(val_before) > 0.01:
                        significant_points.append((x_boundary, val_before, force_type, 'discontinuity'))
        
        # Add zero crossings by checking sign changes between segments
        for i in range(len(segments) - 1):
            x_start = segments[i]
            x_end = segments[i + 1]
            
            if x_end - x_start < 1e-6:
                continue
                
            x_mid = (x_start + x_end) / 2
            N_start, Q_start, M_start = self.internal_forces(x_start + 1e-6)
            N_end, Q_end, M_end = self.internal_forces(x_end - 1e-6)
            
            # Check for zero crossings
            for force_type, val_start, val_end in [('N', N_start, N_end), 
                                                  ('Q', Q_start, Q_end), 
                                                  ('M', M_start, M_end)]:
                if val_start * val_end < 0:  # Sign change indicates zero crossing
                    # Find approximate zero location
                    zero_x = x_start + (x_end - x_start) * abs(val_start) / (abs(val_start) + abs(val_end))
                    significant_points.append((zero_x, 0.0, force_type, 'zero'))
        
        # Filter points to avoid overcrowding
        filtered_points = self.filter_significant_points_robust(significant_points)
        
        # Draw significant values
        for x, value, force_type, point_type in filtered_points:
            # Get world position on beam
            w = self.world_point(x)
            
            # Calculate graph position based on force type
            if force_type == 'N':
                graph_pos = w + self.e_z * value * scale_factor
                color = COLORS['N']
            elif force_type == 'Q':
                graph_pos = w + self.e_z * value * scale_factor
                color = COLORS['Q']
            else:  # M
                # Use smaller scaling for moment display (visual only)
                graph_pos = w + self.e_z * value * scale_factor * 0.1
                color = COLORS['M']
            
            # Format value based on force type
            if force_type == 'M':
                if abs(value) < 1e-6:
                    value_text = "0.0Nm"
                else:
                    value_text = f"{value:.1f}Nm"
            else:
                if abs(value) < 1e-6:
                    value_text = "0.0N"
                else:
                    value_text = f"{value:.1f}N"
            
            # Render text
            text_surface = font_values.render(value_text, True, color)
            
            # Position text with better spacing
            text_offset = self.e_z * (20 if value >= 0 else -25)  # More space, different for positive/negative
            text_pos = graph_pos + text_offset
            
            # Center the text at the position
            text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            
            # Draw small circle at the significant point
            pygame.draw.circle(surf, color, graph_pos.astype(int), 4)  # Slightly larger circle
            
            # Draw the value text
            surf.blit(text_surface, text_rect)

    def filter_significant_points_robust(self, significant_points):
        """More robust filtering for structural analysis diagrams"""
        if not significant_points:
            return []
        
        # Group by force type
        points_by_type = {'N': [], 'Q': [], 'M': []}
        for point in significant_points:
            x, value, force_type, point_type = point
            points_by_type[force_type].append(point)
        
        filtered_points = []
        min_distance = 50  # Minimum distance in pixels
        
        for force_type in ['N', 'Q', 'M']:
            type_points = points_by_type[force_type]
            if not type_points:
                continue
            
            # Sort by x position
            type_points.sort(key=lambda p: p[0])
            
            # Priority system for structural analysis
            priority = {
                'zero': 1,           # Highest priority - always show zeros
                'discontinuity': 2,  # High priority - show jumps
                'maximum': 3,        # Medium priority - show peaks
                'minimum': 3,        # Medium priority - show valleys  
                'constant': 4        # Lower priority - show representative values
            }
            
            # First pass: always keep zeros and major discontinuities
            must_keep = []
            others = []
            
            for point in type_points:
                x, value, force_type, point_type = point
                if point_type in ['zero', 'discontinuity'] or abs(value) > 10:  # Keep significant values
                    must_keep.append(point)
                else:
                    others.append(point)
            
            # Add must-keep points
            filtered_points.extend(must_keep)
            
            # Filter others by distance
            last_x = -float('inf')
            for point in others:
                x, value, force_type, point_type = point
                if x - last_x >= min_distance:
                    filtered_points.append(point)
                    last_x = x
            
            # Limit total points per force type
            type_filtered = [p for p in filtered_points if p[2] == force_type]
            if len(type_filtered) > 6:  # Max 6 points per force type
                # Keep the most important ones
                type_filtered.sort(key=lambda p: (priority.get(p[3], 5), -abs(p[1])))
                # Remove excess points from filtered_points
                filtered_points = [p for p in filtered_points if p[2] != force_type]
                filtered_points.extend(type_filtered[:6])
        
        return filtered_points

def draw_grid(surface):
    """Draws a subtle grid of small crosses at snapping points"""
    w, h = surface.get_size()
    cross_size = 3  # Size of crosses in pixels
    grid_color = COLORS['grid']
    
    # Start from first grid interval (GRID_SIZE) to avoid edge crosses
    for x in range(GRID_SIZE, w, GRID_SIZE):
        for y in range(GRID_SIZE, h, GRID_SIZE):
            # Horizontal line of the cross
            pygame.draw.line(surface, grid_color, 
                           (x - cross_size, y), (x + cross_size, y), 1)
            # Vertical line of the cross
            pygame.draw.line(surface, grid_color, 
                           (x, y - cross_size), (x, y + cross_size), 1)

def draw_slider(surf, x, y, width, value, min_val, max_val, label):
    """Draws a slider control"""
    # Blueish dark-grey background
    slider_bg_color = (45, 55, 75)  # Blueish dark-grey
    
    # Slider background
    pygame.draw.rect(surf, slider_bg_color, (x, y, width, 20), 0)
    
    # Dimmer blue border and knob
    dimmer_blue = (90, 130, 180)  # Less bright than ui_text
    pygame.draw.rect(surf, dimmer_blue, (x, y, width, 20), 2)
    
    # Calculate slider position
    slider_pos = x + (value - min_val) / (max_val - min_val) * width
    
    # Dimmer blue slider handle
    pygame.draw.circle(surf, slider_bg_color, (int(slider_pos), y + 10), 8)
    pygame.draw.circle(surf, dimmer_blue, (int(slider_pos), y + 10), 6)
    
    # Position label and value centered under the slider
    font_slider = get_font('slider')
    label_text = font_slider.render(f"{label}: {value:.1f}", True, COLORS['ui_text'])
    # Center the text under the slider
    text_rect = label_text.get_rect()
    text_x = x + (width - text_rect.width) // 2  # Center horizontally
    surf.blit(label_text, (text_x, y + 25))  # 25 pixels under the slider
    
    return (x, y, width, 20)  # Return for collision detection

def handle_slider_click(mouse_pos, slider_rect, min_val, max_val):
    """Handles clicks on the slider with extended click area"""
    x, y, width, height = slider_rect
    # Extended click area: 10 pixels above and below the slider
    extended_y = y - 10
    extended_height = height + 20
    
    if x <= mouse_pos[0] <= x + width and extended_y <= mouse_pos[1] <= extended_y + extended_height:
        # Calculate new value based on mouse position - bound to right side
        relative_pos = (mouse_pos[0] - x) / width
        # Invert so right side gives max value, left side gives min value
        new_value = min_val + relative_pos * (max_val - min_val)
        return max(min_val, min(max_val, new_value))
    return None

def find_item_under_mouse(mouse_pos, beam, detection_radius=15):
    """Find which item (point load, line load, support, or beam) is under the mouse cursor"""
    if not beam:
        return None, None
    
    # Check point loads first (highest priority)
    for i, (pos_global, force_global) in enumerate(beam.point_loads):
        # Check proximity to the point load start position
        if np.linalg.norm(mouse_pos - pos_global) <= detection_radius:
            return ('point_load', i), pos_global
        
        # Also check along the entire arrow length for easier clicking
        if np.linalg.norm(force_global) > 0:
            arrow_tip = pos_global + force_global
            # Create a line segment from pos_global to arrow_tip and check distance to it
            arrow_vector = force_global
            arrow_length = np.linalg.norm(arrow_vector)
            
            # Vector from arrow start to mouse
            to_mouse = mouse_pos - pos_global
            # Project onto arrow direction
            projection_length = np.dot(to_mouse, arrow_vector) / arrow_length
            projection_length = max(0, min(arrow_length, projection_length))  # Clamp to arrow length
            
            # Find closest point on arrow line
            closest_point = pos_global + (arrow_vector / arrow_length) * projection_length
            
            # Check if mouse is close to the arrow line
            if np.linalg.norm(mouse_pos - closest_point) <= detection_radius:
                return ('point_load', i), pos_global
    
    # Check line loads (check both start and end positions, and the polygon area)
    for i, (start_pos, end_pos, end_amplitude, start_amplitude) in enumerate(beam.line_loads):
        # Check start position
        if np.linalg.norm(mouse_pos - start_pos) <= detection_radius:
            return ('line_load', i), start_pos
        # Check end position  
        if np.linalg.norm(mouse_pos - end_pos) <= detection_radius:
            return ('line_load', i), end_pos
        # Check if mouse is within the line load polygon
        force_vector_end = beam.e_z * end_amplitude
        force_vector_start = beam.e_z * start_amplitude
        polygon_points = [start_pos, end_pos, end_pos + force_vector_end, start_pos + force_vector_start]
        if point_in_polygon(mouse_pos, polygon_points):
            return ('line_load', i), (start_pos + end_pos) / 2
    
    # Check supports
    for support_pos, support_type in beam.supports.items():
        if support_pos == "start":
            support_position = beam.start
        elif support_pos == "end":
            support_position = beam.end
        else:
            continue
        if np.linalg.norm(mouse_pos - support_position) <= detection_radius + 5:  # Slightly larger radius for supports
            return ('support', support_pos), support_position
    
    # Check beam itself (lowest priority)
    # Check if mouse is close to the beam line
    vec_to_point = mouse_pos - beam.start
    projection_length = np.dot(vec_to_point, beam.e_x)
    if 0 <= projection_length <= beam.L:
        closest_point = beam.start + projection_length * beam.e_x
        distance_to_beam = np.linalg.norm(mouse_pos - closest_point)
        if distance_to_beam <= 15:  # Within beam thickness + some margin
            return ('beam', 0), closest_point
    
    return None, None

def point_in_polygon(point, polygon_points):
    """Check if a point is inside a polygon using ray casting algorithm"""
    if len(polygon_points) < 3:
        return False
    
    x, y = point
    n = len(polygon_points)
    inside = False
    
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def delete_item_from_beam(beam, item_type, item_identifier):
    """Delete an item from the beam"""
    if not beam:
        return None
    
    item_category, index_or_pos = item_type, item_identifier
    
    if item_category == 'point_load' and 0 <= index_or_pos < len(beam.point_loads):
        del beam.point_loads[index_or_pos]
        return beam
    elif item_category == 'line_load' and 0 <= index_or_pos < len(beam.line_loads):
        del beam.line_loads[index_or_pos]
        return beam
    elif item_category == 'support' and index_or_pos in beam.supports:
        del beam.supports[index_or_pos]
        return beam
    elif item_category == 'beam':
        # Delete the entire beam (return None to indicate beam should be deleted)
        return None
    
    return beam

beam = None
mode = "idle"
clicks = []
temp_beam = None  # Temporary beam for preview
scale_factor = 0.7  # Scaling factor for internal force diagrams  
slider_dragging = False
show_debug = False  # Debug display on/off
animation_time = 0  # Time for oscillating preview animations
frame_count = 0  # Performance optimization: frame counter for cache management
delete_highlighted_item = None  # Item highlighted for deletion: ('type', index) or ('support', position)
delete_highlighted_pos = None   # Position for visual feedback

def calculate_wave_parameters(force_vector, arrow_head_length, wave_period_length):
    """Calculate common wave parameters for animation - optimized version"""
    force_norm = np.linalg.norm(force_vector)
    if force_norm <= 5:
        return None
    
    force_unit = force_vector / force_norm
    perpendicular = geometry_cache.get_perpendicular_vector(force_unit)  # Use cache
    wave_end_distance = max(5, force_norm - arrow_head_length)
    effective_force_vec = force_vector * (wave_end_distance / force_norm)
    wave_cycles = wave_end_distance / wave_period_length
    
    return {
        'force_norm': force_norm,
        'force_unit': force_unit,
        'perpendicular': perpendicular,
        'wave_end_distance': wave_end_distance,
        'effective_force_vec': effective_force_vec,
        'wave_cycles': wave_cycles
    }

def generate_wave_points(start_pos, wave_params, num_segments, phase_offset, animation_time, wave_frequency, wave_amplitude):
    """Generate points for a wavy line animation"""
    if wave_params is None:
        return []
    
    wave_points = []
    for i in range(num_segments + 1):
        t = i / num_segments
        base_pos = start_pos + t * wave_params['effective_force_vec']
        wave_phase = t * wave_params['wave_cycles'] * 2 * math.pi
        wave_offset = wave_params['perpendicular'] * wave_amplitude * math.sin(
            2 * math.pi * wave_frequency * animation_time + wave_phase + phase_offset
        )
        wave_points.append(base_pos + wave_offset)
    return wave_points

def generate_polygon_edge_points(start_pos, wave_params, num_segments, phase_offset, animation_time, wave_frequency, wave_amplitude):
    """Generate points for polygon edge animation stopping at arrow base"""
    if wave_params is None:
        # Return straight edge points for zero amplitude
        edge_points = []
        for k in range(num_segments + 1):
            edge_t = k / num_segments
            base_pos = start_pos
            edge_points.append(base_pos)
        return edge_points
    
    edge_points = []
    for k in range(num_segments + 1):
        edge_t = k / num_segments
        base_pos = start_pos + edge_t * wave_params['effective_force_vec']
        wave_phase = edge_t * wave_params['wave_cycles'] * 2 * math.pi
        wave_offset = wave_params['perpendicular'] * wave_amplitude * math.sin(
            2 * math.pi * wave_frequency * animation_time + wave_phase + phase_offset
        )
        edge_points.append(base_pos + wave_offset)
    return edge_points

def draw_animated_arrow(screen, arrow_start, force_vector, wave_params, phase_offset, animation_time, 
                       wave_frequency, wave_amplitude, num_segments, arrow_head_length, color):
    """Draw a single animated arrow with wavy line"""
    if wave_params is None:
        # Fallback to simple line for very short arrows
        arrow_end = arrow_start + force_vector
        pygame.draw.line(screen, color, arrow_start, arrow_end, FORCE_LINE_THICKNESS)
        return
    
    # Draw wavy line
    wave_points = generate_wave_points(arrow_start, wave_params, num_segments, phase_offset, 
                                     animation_time, wave_frequency, wave_amplitude)
    if len(wave_points) > 1:
        pygame.draw.lines(screen, color, False, wave_points, FORCE_LINE_THICKNESS)
    
    # Calculate animated positions
    arrow_end = arrow_start + force_vector
    wave_end_point = arrow_start + wave_params['effective_force_vec']
    end_wave_phase = wave_params['wave_cycles'] * 2 * math.pi
    wave_connection_offset = wave_params['perpendicular'] * wave_amplitude * math.sin(
        2 * math.pi * wave_frequency * animation_time + end_wave_phase + phase_offset
    )
    animated_wave_end = wave_end_point + wave_connection_offset
    animated_tip = arrow_end + wave_connection_offset
    
    # Draw connecting line and animated arrowhead
    pygame.draw.line(screen, color, animated_wave_end, animated_tip, FORCE_LINE_THICKNESS)
    
    # Optimized arrowhead using geometry cache
    triangle_width = arrow_head_length * ARROW_SIZE_RATIO
    arrow_points = geometry_cache.get_arrow_points(
        animated_tip, wave_params['force_unit'], arrow_head_length, triangle_width
    )
    pygame.draw.polygon(screen, color, arrow_points)

def create_animated_polygon(clicks, wave_params_left, wave_params_right, num_segments, animation_time,
                           wave_frequency, wave_amplitude, force_vector_start=None, force_vector_end=None):
    """Create polygon with animated side edges, stopping at arrow bases without top edge line"""
    if force_vector_start is None:
        force_vector_start = wave_params_left['effective_force_vec'] if wave_params_left else np.array([0, 0])
    if force_vector_end is None:
        force_vector_end = wave_params_right['effective_force_vec'] if wave_params_right else np.array([0, 0])
    
    # Calculate number of arrows using line load spacing
    line_length = np.linalg.norm(clicks[1] - clicks[0])
    if line_length < LINE_LOAD_SPACING * 2:
        num_arrows = 2
    else:
        # Calculate number of arrows to fit evenly with LINE_LOAD_SPACING
        num_arrows = int(line_length / LINE_LOAD_SPACING) + 1  # +1 to include both endpoints
        num_arrows = max(2, num_arrows)  # At least 2 arrows (start and end)
    
    # Generate edge points with synchronized phase offsets, stopping at arrow bases
    left_edge_points = generate_polygon_edge_points(
        clicks[0], wave_params_left, num_segments, 0.0, animation_time, wave_frequency, wave_amplitude
    )
    right_edge_points = generate_polygon_edge_points(
        clicks[1], wave_params_right, num_segments, (num_arrows - 1) * 0.3,
        animation_time, wave_frequency, wave_amplitude
    )
    
    # Create simple polygon: bottom edge + right wavy edge + straight top edge + left wavy edge (reversed)
    animated_polygon_points = (
        [clicks[0], clicks[1]] +  # Bottom edge (straight)
        right_edge_points[1:] +   # Right edge (wavy, skip first point to avoid duplicate)
        list(reversed(left_edge_points[1:]))  # Left edge (wavy, reversed, skip first point)
    )
    return animated_polygon_points

running = True
while running:
    # Performance optimization: clear geometry cache every 300 frames to prevent memory buildup
    frame_count += 1
    if frame_count % 300 == 0:
        geometry_cache.clear_cache()
    
    screen.fill(COLORS['bg'])
    draw_grid(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = snap(pygame.mouse.get_pos())
            
            # Check slider interaction first only if statically determinate
            if beam:
                is_determinate, _ = beam.check_static_determinacy()
                if is_determinate:
                    slider_rect = (screen.get_width() - 220, 10, 200, 20)
                    new_scale = handle_slider_click(pygame.mouse.get_pos(), slider_rect, 0.1, 2.0)
                    if new_scale is not None:
                        scale_factor = new_scale
                        slider_dragging = True
                        continue

            # Handle delete mode
            if mode == "delete" and beam:
                if delete_highlighted_item:
                    item_type, item_identifier = delete_highlighted_item
                    result_beam = delete_item_from_beam(beam, item_type, item_identifier)
                    if result_beam is None:
                        # Beam was deleted - reset everything
                        beam = None
                        temp_beam = None
                        mode = "idle"
                        clicks = []
                    else:
                        beam = result_beam
                    delete_highlighted_item = None
                    delete_highlighted_pos = None
                continue

            if mode == "idle":
                clicks = [pos]
                mode = "beam"
            elif mode == "beam":
                clicks.append(pos)
                if len(clicks) >= 2 and np.linalg.norm(clicks[1] - clicks[0]) > 5:
                    beam = Beam(clicks[0], clicks[1])
                    temp_beam = None
                    mode = "idle"
                    clicks = []
            elif mode == "point_load":
                if len(clicks) == 0:
                    # First click: snap to beam for consistent display
                    if beam:
                        snapped_pos = beam.snap_to_beam(pos)
                        clicks = [snapped_pos]
                elif len(clicks) == 1:
                    # Second click: add the point load
                    if beam:
                        snapped_pos = beam.snap_to_beam(clicks[0])
                        beam.add_point_load(snapped_pos, pos - snapped_pos)
                    # Reset for next point load
                    clicks = []
            elif mode == "line_load":
                if len(clicks) == 0:
                    # First click: snap to beam
                    if beam:
                        snapped_pos = beam.snap_to_beam(pos)
                        clicks = [snapped_pos]
                elif len(clicks) == 1:
                    # Second click: snap to beam - only along beam axis
                    if beam:
                        # Calculate the projection of the click onto the beam axis
                        vec_to_click = pos - beam.start
                        projection_length = np.dot(vec_to_click, beam.e_x)
                        # Limit to beam length
                        projection_length = max(0, min(beam.L, projection_length))
                        # The second point must lie on the beam
                        snapped_pos = beam.start + projection_length * beam.e_x
                        clicks.append(snapped_pos)
                elif len(clicks) == 2:
                    # Third click: set uniform amplitude
                    if beam:
                        mid = 0.5 * (clicks[0] + clicks[1])
                        direction = pos - mid
                        # Create uniform line load with same amplitude at both ends
                        beam.add_line_load(clicks[0], clicks[1], direction, direction)
                    # Reset for next line load
                    clicks = []
            elif mode == "trapezoidal_load":
                if len(clicks) == 0:
                    # First click: snap to beam
                    if beam:
                        snapped_pos = beam.snap_to_beam(pos)
                        clicks = [snapped_pos]
                elif len(clicks) == 1:
                    # Second click: snap to beam - only along beam axis allowed
                    if beam:
                        # Calculate the projection of the click onto the beam axis
                        vec_to_click = pos - beam.start
                        projection_length = np.dot(vec_to_click, beam.e_x)
                        # Limit to beam length
                        projection_length = max(0, min(beam.L, projection_length))
                        # Second point must lie on the beam
                        snapped_pos = beam.start + projection_length * beam.e_x
                        clicks.append(snapped_pos)
                elif len(clicks) == 2:
                    # Third click: set end amplitude direction
                    if beam:
                        direction = pos - 0.5 * (clicks[0] + clicks[1])
                        clicks.append(direction)  # Store end amplitude direction
                elif len(clicks) == 3:
                    # Fourth click: set start amplitude and create load
                    if beam:
                        end_direction = clicks[2]
                        start_direction = pos - 0.5 * (clicks[0] + clicks[1])
                        beam.add_line_load(clicks[0], clicks[1], end_direction, start_direction)
                    # Reset for next trapezoidal load
                    clicks = []
            elif mode == "support":
                if beam:
                    snap_pos = beam.add_support(pos)
                # Stay in support mode for more support changes
                clicks = []

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            slider_dragging = False
            
        elif event.type == pygame.MOUSEMOTION and slider_dragging:
            # Update slider during dragging only if statically determinate
            if beam:
                is_determinate, _ = beam.check_static_determinacy()
                if is_determinate:
                    slider_rect = (screen.get_width() - 220, 10, 200, 20)
                    new_scale = handle_slider_click(pygame.mouse.get_pos(), slider_rect, 0.1, 2.0)
                    if new_scale is not None:
                        scale_factor = new_scale

        elif event.type == pygame.MOUSEMOTION:
            # Handle delete mode highlighting
            if mode == "delete" and beam:
                mouse_pos = snap(pygame.mouse.get_pos())
                delete_highlighted_item, delete_highlighted_pos = find_item_under_mouse(mouse_pos, beam)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # Aktion abbrechen
                mode = "idle"
                clicks = []
                temp_beam = None
                delete_highlighted_item = None
                delete_highlighted_pos = None
            elif event.key == pygame.K_b:
                mode = "beam"
                clicks = []
                temp_beam = None
                delete_highlighted_item = None
                delete_highlighted_pos = None
            elif event.key == pygame.K_p:
                if beam:
                    mode = "point_load"
                    clicks = []
                    delete_highlighted_item = None
                    delete_highlighted_pos = None
            elif event.key == pygame.K_l:
                # L key: Simple uniform line load (3 clicks)
                if beam:
                    mode = "line_load"
                    clicks = []
                    delete_highlighted_item = None
                    delete_highlighted_pos = None
            elif event.key == pygame.K_t:
                # T key: Trapezoidal/variable line load (4 clicks)
                if beam:
                    mode = "trapezoidal_load"
                    clicks = []
                    delete_highlighted_item = None
                    delete_highlighted_pos = None
            elif event.key == pygame.K_s:
                if beam:
                    mode = "support"
                    clicks = []
                    delete_highlighted_item = None
                    delete_highlighted_pos = None
            elif event.key == pygame.K_c:
                beam = None
                temp_beam = None
                mode = "idle"
                clicks = []
                delete_highlighted_item = None
                delete_highlighted_pos = None
            elif event.key == pygame.K_d and (event.mod & pygame.KMOD_SHIFT):
                # Shift+D: Debug display toggle
                show_debug = not show_debug
            elif event.key == pygame.K_d:
                # D key: Delete mode
                if beam and mode != "delete":
                    mode = "delete"
                    clicks = []
                elif mode == "delete":
                    mode = "idle"
                    clicks = []
                    delete_highlighted_item = None
                    delete_highlighted_pos = None

    # Draw beam (finished or in progress)
    if beam:
        beam.draw(screen, delete_highlighted_item)
        beam.draw_diagrams(screen, scale_factor)
        if show_debug:
            beam.debug_internal_forces(screen)  # Show debug info

    # Vorschau während der Erstellung
    mpos = snap(pygame.mouse.get_pos())
    
    if mode == "beam" and len(clicks) == 1:
        # Beam preview with length display
        if np.linalg.norm(mpos - clicks[0]) > 5:
            temp_beam = Beam(clicks[0], mpos)
            temp_beam.draw(screen)
            
            # Calculate and display length
            beam_length_pixels = np.linalg.norm(mpos - clicks[0])
            beam_length_meters = beam_length_pixels / GRID_SIZE  # Convert from pixels to meters
            
            # Position text like with point loads
            font_preview = get_font('preview')
            length_text = font_preview.render(f"{beam_length_meters:.1f}m", True, COLORS['beam'])  # Match beam color
            
            # Position text in the middle of the beam, slightly above
            mid_point = (clicks[0] + mpos) / 2
            text_pos = mid_point + np.array([0, -30])  # 30 pixels above beam center
            text_rect = length_text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            
            screen.blit(length_text, text_rect)
    
    elif mode == "point_load" and len(clicks) == 1:
        # Point load preview with oscillating sine wave animation
        tip = mpos  # Mouse position is the tip
        start = clicks[0]  # Start point on beam
        
        # Calculate arrow direction
        force_vec = tip - start
        force_norm = np.linalg.norm(force_vec)
        
        if force_norm > 5:  # Only draw if mouse is far enough from start
            # Calculate wave parameters for point load (uses more segments)
            point_load_segments = ANIMATION_SEGMENTS + 20
            wave_params = {
                'force_norm': force_norm,
                'force_unit': force_vec / force_norm,
                'perpendicular': np.array([-force_vec[1], force_vec[0]]) / force_norm,
                'wave_end_distance': max(10, force_norm - ARROW_HEAD_SIZE),
                'effective_force_vec': force_vec * (max(10, force_norm - ARROW_HEAD_SIZE) / force_norm),
                'wave_cycles': max(10, force_norm - ARROW_HEAD_SIZE) / ANIMATION_PERIOD_LENGTH
            }
            
            # Draw wavy line
            wave_points = generate_wave_points(start, wave_params, point_load_segments, 0, 
                                             animation_time, ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE)
            if len(wave_points) > 1:
                pygame.draw.lines(screen, COLORS['force_preview'], False, wave_points, FORCE_LINE_THICKNESS)
            
            # Calculate animated positions
            wave_end_point = start + wave_params['effective_force_vec']
            end_wave_phase = wave_params['wave_cycles'] * 2 * math.pi
            wave_connection_offset = wave_params['perpendicular'] * ANIMATION_AMPLITUDE * math.sin(
                2 * math.pi * ANIMATION_FREQUENCY * animation_time + end_wave_phase
            )
            animated_wave_end = wave_end_point + wave_connection_offset
            animated_tip = tip + wave_connection_offset
            
            # Draw connecting line and arrowhead
            pygame.draw.line(screen, COLORS['force_preview'], animated_wave_end, animated_tip, FORCE_LINE_THICKNESS)
            
            # Animated arrowhead
            force_unit = wave_params['force_unit']
            perpendicular = wave_params['perpendicular']
            triangle_width = ARROW_HEAD_SIZE * ARROW_SIZE_RATIO  # Use variables for triangle base
            
            # Triangle points: tip at animated position, base centered on line
            base_center = animated_tip - force_unit * ARROW_HEAD_SIZE
            left_base = base_center - perpendicular * triangle_width
            right_base = base_center + perpendicular * triangle_width
            
            arrow_points = [animated_tip, left_base, right_base]
            pygame.draw.polygon(screen, COLORS['force_preview'], arrow_points)
            
            # Display load intensity - STATIC position
            load_intensity = f"{force_norm:.0f}N"
            font_preview = get_font('preview')
            text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
            text_offset = force_unit * 25
            text_pos = tip + text_offset
            text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            screen.blit(text_surface, text_rect)
        
    elif mode == "line_load" and len(clicks) == 1:
        # Simple line load start preview - only along beam axis
        if beam:
            # Calculate projection of mouse position onto beam axis
            vec_to_mouse = mpos - beam.start  
            projection_length = np.dot(vec_to_mouse, beam.e_x)
            # Limit to beam length
            projection_length = max(0, min(beam.L, projection_length))
            # Show preview only along beam with normal line thickness
            preview_end = beam.start + projection_length * beam.e_x
            pygame.draw.line(screen, COLORS['force_preview'], clicks[0], preview_end, FORCE_LINE_THICKNESS)
        
    elif mode == "line_load" and len(clicks) == 2:
        # Simple line load uniform preview with oscillating animation
        if beam:
            mid = 0.5 * (clicks[0] + clicks[1])
            mouse_vec = mpos - mid
            amplitude = np.dot(mouse_vec, beam.e_z)
            force_vector = beam.e_z * amplitude
            
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            
            # If the line load is shorter than minimum spacing, only show edge arrows
            if line_length < LINE_LOAD_SPACING * 2:  # Need at least 2x spacing for 2 arrows
                num_arrows = 2  # Only edge arrows
            else:
                # Calculate number of arrows to fit evenly with LINE_LOAD_SPACING
                num_arrows = int(line_length / LINE_LOAD_SPACING) + 1  # +1 to include both endpoints
                num_arrows = max(2, num_arrows)  # At least 2 arrows (start and end)
            
            # Calculate wave parameters
            wave_params = calculate_wave_parameters(force_vector, ANIMATION_ARROW_HEAD_LENGTH, ANIMATION_PERIOD_LENGTH)
            
            # Create animated polygon
            if wave_params:
                animated_polygon_points = create_animated_polygon(
                    clicks, wave_params, wave_params, ANIMATION_SEGMENTS, animation_time,
                    ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, force_vector, force_vector
                )
                draw_transparent_polygon(screen, COLORS['force_preview'], animated_polygon_points, 180)
            else:
                # Static polygon for very short arrows
                rect_points = [clicks[0], clicks[1], clicks[1] + force_vector, clicks[0] + force_vector]
                draw_transparent_polygon(screen, COLORS['force_preview'], rect_points, 180)
            
            # Draw animated arrows
            for i in range(num_arrows):
                t = i / (num_arrows - 1) if num_arrows > 1 else 0
                arrow_start = clicks[0] + t * (clicks[1] - clicks[0])
                phase_offset = i * 0.3
                
                draw_animated_arrow(screen, arrow_start, force_vector, wave_params, phase_offset,
                                  animation_time, ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, 
                                  ANIMATION_SEGMENTS, ANIMATION_ARROW_HEAD_LENGTH, COLORS['force_preview'])
            
            # Display load intensity
            if num_arrows > 0 and np.linalg.norm(force_vector) > 5:
                line_length = np.linalg.norm(clicks[1] - clicks[0])
                load_intensity_per_meter = np.linalg.norm(force_vector) / line_length * 1000 if line_length > 0 else 0
                load_intensity = f"{load_intensity_per_meter:.0f}N/m (uniform)"
                font_preview = get_font('preview')
                text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
                
                force_norm = np.linalg.norm(force_vector)
                force_unit = force_vector / force_norm
                mid_point = (clicks[0] + clicks[1]) / 2
                arrow_end = mid_point + force_vector
                text_offset = force_unit * 25
                text_pos = arrow_end + text_offset
                text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                screen.blit(text_surface, text_rect)
        
    elif mode == "trapezoidal_load" and len(clicks) == 1:
        # Trapezoidal load start preview - only along beam axis
        if beam:
            # Calculate projection of mouse position onto beam axis
            vec_to_mouse = mpos - beam.start  
            projection_length = np.dot(vec_to_mouse, beam.e_x)
            # Limit to beam length
            projection_length = max(0, min(beam.L, projection_length))
            # Show preview only along beam with normal line thickness
            preview_end = beam.start + projection_length * beam.e_x
            pygame.draw.line(screen, COLORS['force_preview'], clicks[0], preview_end, FORCE_LINE_THICKNESS)
        
    elif mode == "trapezoidal_load" and len(clicks) == 2:
        # Trapezoidal load direction preview (end amplitude) with oscillating animation
        if beam:
            mid = 0.5 * (clicks[0] + clicks[1])
            mouse_vec = mpos - mid
            amplitude = np.dot(mouse_vec, beam.e_z)
            force_vector = beam.e_z * amplitude
            
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            
            # If the line load is shorter than minimum spacing, only show edge arrows
            if line_length < LINE_LOAD_SPACING * 2:  # Need at least 2x spacing for 2 arrows
                num_arrows = 2  # Only edge arrows
            else:
                # Calculate number of arrows to fit evenly with LINE_LOAD_SPACING
                num_arrows = int(line_length / LINE_LOAD_SPACING) + 1  # +1 to include both endpoints
                num_arrows = max(2, num_arrows)  # At least 2 arrows (start and end)
            
            # Calculate wave parameters
            wave_params = calculate_wave_parameters(force_vector, ANIMATION_ARROW_HEAD_LENGTH, ANIMATION_PERIOD_LENGTH)
            
            # Create animated polygon
            if wave_params:
                animated_polygon_points = create_animated_polygon(
                    clicks, wave_params, wave_params, ANIMATION_SEGMENTS, animation_time,
                    ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, force_vector, force_vector
                )
                draw_transparent_polygon(screen, COLORS['force_preview'], animated_polygon_points, 180)
            else:
                # Static polygon for very short arrows
                rect_points = [clicks[0], clicks[1], clicks[1] + force_vector, clicks[0] + force_vector]
                draw_transparent_polygon(screen, COLORS['force_preview'], rect_points, 180)
            
            # Draw animated arrows
            for i in range(num_arrows):
                t = i / (num_arrows - 1) if num_arrows > 1 else 0
                arrow_start = clicks[0] + t * (clicks[1] - clicks[0])
                phase_offset = i * 0.3
                
                draw_animated_arrow(screen, arrow_start, force_vector, wave_params, phase_offset,
                                  animation_time, ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, 
                                  ANIMATION_SEGMENTS, ANIMATION_ARROW_HEAD_LENGTH, COLORS['force_preview'])
            
            # Display load intensity
            if num_arrows > 0 and np.linalg.norm(force_vector) > 5:
                line_length = np.linalg.norm(clicks[1] - clicks[0])
                load_intensity_per_meter = np.linalg.norm(force_vector) / line_length * 1000 if line_length > 0 else 0
                load_intensity = f"{load_intensity_per_meter:.0f}N/m (end)"
                font_preview = get_font('preview')
                text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
                
                force_norm = np.linalg.norm(force_vector)
                force_unit = force_vector / force_norm
                mid_point = (clicks[0] + clicks[1]) / 2
                arrow_end = mid_point + force_vector
                text_offset = force_unit * 25
                text_pos = arrow_end + text_offset
                text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                screen.blit(text_surface, text_rect)
    
    elif mode == "trapezoidal_load" and len(clicks) == 3:
        # Streckenlast variable Vorschau - zeige Trapez mit verschiedenen Amplituden und Animation
        if beam:
            mid = 0.5 * (clicks[0] + clicks[1])
            
            # End amplitude (from previous click)
            end_amplitude = np.dot(clicks[2], beam.e_z)
            force_vector_end = beam.e_z * end_amplitude
            
            # Start amplitude (current mouse position)
            mouse_vec = mpos - mid
            start_amplitude = np.dot(mouse_vec, beam.e_z)
            force_vector_start = beam.e_z * start_amplitude
            
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            
            # If the line load is shorter than minimum spacing, only show edge arrows
            if line_length < LINE_LOAD_SPACING * 2:  # Need at least 2x spacing for 2 arrows
                num_arrows = 2  # Only edge arrows
            else:
                # Calculate number of arrows to fit evenly with LINE_LOAD_SPACING
                num_arrows = int(line_length / LINE_LOAD_SPACING) + 1  # +1 to include both endpoints
                num_arrows = max(2, num_arrows)  # At least 2 arrows (start and end)
            
            # Calculate wave parameters for both sides
            wave_params_start = calculate_wave_parameters(force_vector_start, ANIMATION_ARROW_HEAD_LENGTH, ANIMATION_PERIOD_LENGTH)
            wave_params_end = calculate_wave_parameters(force_vector_end, ANIMATION_ARROW_HEAD_LENGTH, ANIMATION_PERIOD_LENGTH)
            
            # Create animated trapezoidal polygon with different edge parameters
            if wave_params_start or wave_params_end:
                animated_polygon_points = create_animated_polygon(
                    clicks, wave_params_start, wave_params_end, ANIMATION_SEGMENTS, animation_time,
                    ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, force_vector_start, force_vector_end
                )
                draw_transparent_polygon(screen, COLORS['force_preview'], animated_polygon_points, 180)
            else:
                # Static polygon for very short arrows
                polygon_points = [clicks[0], clicks[1], clicks[1] + force_vector_end, clicks[0] + force_vector_start]
                draw_transparent_polygon(screen, COLORS['force_preview'], polygon_points, 180)
            
            # Draw variable preview arrows with different lengths and oscillating animation
            for i in range(num_arrows):
                t = i / (num_arrows - 1) if num_arrows > 1 else 0
                arrow_start = clicks[0] + t * (clicks[1] - clicks[0])
                
                # Linear interpolation of amplitude and force vector
                current_amplitude = start_amplitude + t * (end_amplitude - start_amplitude)
                force_vector = beam.e_z * current_amplitude
                
                # Calculate wave parameters for this arrow
                wave_params = calculate_wave_parameters(force_vector, ANIMATION_ARROW_HEAD_LENGTH, ANIMATION_PERIOD_LENGTH)
                phase_offset = i * 0.3
                
                draw_animated_arrow(screen, arrow_start, force_vector, wave_params, phase_offset,
                                  animation_time, ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, 
                                  ANIMATION_SEGMENTS, ANIMATION_ARROW_HEAD_LENGTH, COLORS['force_preview'])
            
            # Display both start and end intensities
            font_preview = get_font('preview')
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            
            # Start intensity
            if line_length > 0:
                start_intensity = abs(start_amplitude) / line_length * 1000
                start_text = f"{start_intensity:.0f}N/m (start)"
                text_surface = font_preview.render(start_text, True, COLORS['force_text'])
                
                force_unit = force_vector_start / np.linalg.norm(force_vector_start) if np.linalg.norm(force_vector_start) > 0 else np.array([0, -1])
                text_offset = force_unit * 25
                text_pos = clicks[0] + force_vector_start + text_offset
                text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                screen.blit(text_surface, text_rect)
            
            # End intensity
            if line_length > 0:
                end_intensity = abs(end_amplitude) / line_length * 1000
                end_text = f"{end_intensity:.0f}N/m (end)"
                text_surface = font_preview.render(end_text, True, COLORS['force_text'])
                
                force_unit = force_vector_end / np.linalg.norm(force_vector_end) if np.linalg.norm(force_vector_end) > 0 else np.array([0, -1])
                text_offset = force_unit * 25
                text_pos = clicks[1] + force_vector_end + text_offset
                text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                screen.blit(text_surface, text_rect)

    # Draw scale slider only when statically determinate
    if beam:
        is_determinate, _ = beam.check_static_determinacy()
        if is_determinate:
            # Position slider in upper right corner - same size, wider range
            slider_rect = draw_slider(screen, screen.get_width() - 220, 10, 200, scale_factor, 0.1, 2.0, "Graph Scale")

    # Hinweistext - immer anzeigen
    font_ui = get_font('ui')
    
    # Shortcuts in zwei Zeilen anzeigen - oben links
    shortcuts_line1 = "B: Beam | P: Point Load | L: Line Load | S: Support"
    shortcuts_line2 = "T: Trapezoidal Load | D: Delete | C: Clear | ESC: Cancel"
    
    # Highlight active function
    active_shortcuts = {
        "beam": "B: Beam",
        "point_load": "P: Point Load",
        "line_load": "L: Line Load",
        "support": "S: Support",
        "trapezoidal_load": "T: Trapezoidal Load",
        "delete": "D: Delete"
    }
    
    # Render shortcuts with highlighting
    def render_highlighted_shortcuts(line, y_pos):
        x_offset = 10
        parts = line.split(" | ")
        for i, part in enumerate(parts):
            if i > 0:
                separator = font_ui.render(" | ", True, COLORS['ui_text'])
                screen.blit(separator, (x_offset, y_pos))
                x_offset += separator.get_width()
            
            # Check if this part should be highlighted
            is_active = mode in active_shortcuts and part.startswith(active_shortcuts[mode].split(":")[0] + ":")
            color = COLORS['ui_highlight'] if is_active else COLORS['ui_text']
            part_text = font_ui.render(part, True, color)
            screen.blit(part_text, (x_offset, y_pos))
            x_offset += part_text.get_width()
    
    render_highlighted_shortcuts(shortcuts_line1, 10)
    render_highlighted_shortcuts(shortcuts_line2, 35)
    
    # Show static determinacy bottom left above system dialogs
    if beam:
        is_determinate, status_text = beam.check_static_determinacy()
        status_color = COLORS['status_ok'] if is_determinate else COLORS['status_error']
        status_display = font_ui.render(status_text, True, status_color)
        screen.blit(status_display, (10, screen.get_height() - 55))  # 25 pixels above system dialogs
    
    # Status display based on mode (at bottom of screen)
    if mode != "idle":
        if mode == "beam":
            if len(clicks) == 0:
                msg = "Beam: Click start point"
            else:
                msg = "Beam: Click end point"
        elif mode == "point_load":
            if len(clicks) == 0:
                msg = "Point Load: Click application point"
            else:
                msg = "Point Load: Set force direction and magnitude"
        elif mode == "line_load":
            if len(clicks) == 0:
                msg = "Uniform Line Load: Click start point"
            elif len(clicks) == 1:
                msg = "Uniform Line Load: Click end point"
            else:
                msg = "Uniform Line Load: Set uniform force magnitude (perpendicular)"
        elif mode == "trapezoidal_load":
            if len(clicks) == 0:
                msg = "Trapezoidal Load: Click start point"
            elif len(clicks) == 1:
                msg = "Trapezoidal Load: Click end point"
            elif len(clicks) == 2:
                msg = "Trapezoidal Load: Set force magnitude at START (perpendicular)"
            else:
                msg = "Trapezoidal Load: Set force magnitude at END (for variable load)"
        elif mode == "support":
            msg = "Support: Click beam end (multiple clicks to change type)"
        elif mode == "delete":
            msg = "Delete: Click on beam, load, or support to delete"
        else:
            msg = f"Mode: {mode}"
        
        status_text = font_ui.render(msg, True, COLORS['ui_highlight'])
        screen.blit(status_text, (10, screen.get_height() - 30))

    # Update animation time for oscillating preview effects
    animation_time += clock.get_time() / 1000.0  # Convert milliseconds to seconds

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()