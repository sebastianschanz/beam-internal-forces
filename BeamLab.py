import pygame
import numpy as np
import sys

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("BeamLab - Structural Analysis App")
clock = pygame.time.Clock()

# Color System - Clean and Organized
COLORS = {
    # Background and Grid
    'bg': (30, 30, 40),
    'grid': (60, 60, 70),
    'c2025': (100, 100, 120),
    
    # UI Elements
    'ui_text': (90, 150, 220),
    'ui_highlight': (150, 230, 255),
    'status_error': (255, 255, 60),
    'slider_bg_color': (45, 55, 75),
    'slider_mark_color': (65, 75, 95),
    
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
EDGE_MARGIN = 15
CLICK_PROXIMITY = 10  # Proximity for click detection

# Font configuration - Consolidated to reduce redundancy
MAIN_FONT = 'consolas'
SMALL_FONT_SIZE = 14
MEDIUM_FONT_SIZE = 18
LARGE_FONT_SIZE = 28

# Slider configuration
SLIDER_MIN = 0.1  # Minimum value for sliders
SLIDER_MAX = 2.0  # Maximum value for sliders
SLIDER_WIDTH = 175  # Width of sliders
SLIDER_THICKNESS = 20 # Thickness of sliders

# Graphics configuration
BEAM_THICKNESS = 10  # Thickness of beams in pixels

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

FONTS = {
    'axis': (MAIN_FONT, SMALL_FONT_SIZE),
    'values': (MAIN_FONT, SMALL_FONT_SIZE),
    'reactions': (MAIN_FONT, SMALL_FONT_SIZE),
    'legend': (MAIN_FONT, SMALL_FONT_SIZE),
    'slider': (MAIN_FONT, SMALL_FONT_SIZE),
    'ui': (MAIN_FONT, MEDIUM_FONT_SIZE),
    'preview': (MAIN_FONT, LARGE_FONT_SIZE)
}

# Performance optimization: Pre-cache fonts
_font_cache = {}
def get_font(font_key):
    """Get font by key with caching for performance"""
    if font_key not in _font_cache:
        family, size = FONTS[font_key]
        _font_cache[font_key] = pygame.font.SysFont(family, size)
    return _font_cache[font_key]

# Geometry cache for performance optimization
class GeometryCache:
    def __init__(self):
        self._perpendicular_cache = {}
        self._arrow_cache = {}
        self._distance_cache = {}
    
    def get_perpendicular_vector(self, vector):
        """Get perpendicular vector with caching"""
        key = (round(vector[0], 6), round(vector[1], 6))  # Round to avoid floating point issues
        if key not in self._perpendicular_cache:
            self._perpendicular_cache[key] = np.array([-vector[1], vector[0]])
        return self._perpendicular_cache[key]
    
    def get_arrow_points(self, tip, direction_unit, arrow_length, arrow_width):
        """Generate arrow triangle points with caching"""
        key = (tuple(np.round(tip, 2)), tuple(np.round(direction_unit, 6)), arrow_length, arrow_width)
        if key not in self._arrow_cache:
            base = tip - direction_unit * arrow_length
            perp = self.get_perpendicular_vector(direction_unit) * arrow_width
            self._arrow_cache[key] = [tip, base + perp, base - perp]
        return self._arrow_cache[key]
    
    def get_distance(self, p1, p2):
        """Calculate distance between two points with caching"""
        key = (tuple(np.round(p1, 2)), tuple(np.round(p2, 2)))
        if key not in self._distance_cache:
            self._distance_cache[key] = np.linalg.norm(np.array(p2) - np.array(p1))
        return self._distance_cache[key]
    
    def clear_cache(self):
        """Clear caches when they get too large"""
        if len(self._perpendicular_cache) > 1000:
            self._perpendicular_cache.clear()
        if len(self._arrow_cache) > 1000:
            self._arrow_cache.clear()
        if len(self._distance_cache) > 1000:
            self._distance_cache.clear()

# Performance optimization: Mouse position cache
class MouseCache:
    def __init__(self):
        self._last_pos = None
        self._last_snapped_pos = None
        self._frame_count = 0
    
    def get_mouse_pos(self):
        """Get current mouse position with caching"""
        current_pos = pygame.mouse.get_pos()
        if self._last_pos != current_pos:
            self._last_pos = current_pos
            self._last_snapped_pos = None  # Reset snapped position when mouse moves
        return current_pos
    
    def get_snapped_mouse_pos(self):
        """Get snapped mouse position with caching"""
        current_pos = self.get_mouse_pos()
        if self._last_snapped_pos is None or self._last_pos != current_pos:
            self._last_snapped_pos = snap(current_pos)
        return self._last_snapped_pos

# Explosion Animation System
class ExplosionParticle:
    def __init__(self, pos, velocity, color, size, lifetime, particle_type='circle'):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.initial_size = size
        self.size = size
        self.lifetime = lifetime
        self.age = 0.0
        self.gravity = np.array([0, 600])  # Much faster downward gravity
        self.particle_type = particle_type
        self.rotation = 0.0
        self.rotation_speed = np.random.uniform(-15, 15)  # Faster rotation for beam chunks
        
    def update(self, dt):
        """Update particle position and properties"""
        self.age += dt
        if self.age >= self.lifetime:
            return False  # Particle is dead
            
        # Update position with velocity and gravity
        self.velocity += self.gravity * dt
        self.pos += self.velocity * dt
        
        # Update rotation for beam chunks
        self.rotation += self.rotation_speed * dt
        
        # Update size (shrink over time, but slower for beam chunks)
        progress = self.age / self.lifetime
        if self.particle_type == 'beam_chunk':
            self.size = self.initial_size * (1.0 - progress * 0.3)  # Beam chunks shrink less
        else:
            self.size = self.initial_size * (1.0 - progress * 0.7)  # Regular particles shrink more
        
        return True  # Particle is alive
        
    def draw(self, surf):
        """Draw the particle"""
        if self.age >= self.lifetime:
            return
            
        if self.particle_type == 'beam_chunk':
            # Draw rectangular beam chunks with alpha blending, no contours
            if self.size > 1:
                chunk_width = int(self.size * 2.0)
                chunk_height = int(self.size * 1.2)
                temp_surface = pygame.Surface((chunk_width * 2, chunk_height * 2), pygame.SRCALPHA)
                progress = self.age / self.lifetime
                alpha = int(255 * (1.0 - progress))
                color_with_alpha = (*self.color, alpha)
                rect = pygame.Rect(chunk_width // 2, chunk_height // 2, chunk_width, chunk_height)
                pygame.draw.rect(temp_surface, color_with_alpha, rect)
                rotated_surface = pygame.transform.rotate(temp_surface, self.rotation)
                rotated_rect = rotated_surface.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
                surf.blit(rotated_surface, rotated_rect)
        else:
            # Draw particle as a circle (fire/smoke) with fade alpha over time
            progress = self.age / self.lifetime
            alpha = int(255 * (1.0 - progress))
            
            # Create color with alpha for fire particles only
            color_with_alpha = (*self.color, alpha)
            
            if self.size > 1:
                temp_surface = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surface, color_with_alpha, 
                                 (int(self.size), int(self.size)), int(self.size))
                surf.blit(temp_surface, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))

class ExplosionSystem:
    def __init__(self):
        self.explosions = []  # List of active explosions
        
    def create_explosion(self, center_pos, intensity=50, beam_color=None):
        """Create an explosion at the given position with optional beam chunks"""
        particles = []
        
        # Create beam chunks if beam_color is provided
        if beam_color is not None:
            # Create 5-8 bigger beam chunks
            num_chunks = np.random.randint(5, 9)
            for i in range(num_chunks):
                # Random angle and slower speed for chunks (slower than sparks)
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(150, 300)  # Slower speed for chunks
                velocity = np.array([np.cos(angle) * speed, np.sin(angle) * speed])
                
                # Vary beam chunk color slightly for more realism
                color_variation = np.random.randint(-20, 21)
                varied_color = tuple(max(0, min(255, c + color_variation)) for c in beam_color)
                
                # Much bigger beam chunk size and shorter lifetime
                size = np.random.uniform(15, 25)  # Much bigger chunks
                lifetime = np.random.uniform(0.3, 0.8)  # Much shorter lifetime for chunks
                
                particle = ExplosionParticle(center_pos, velocity, varied_color, size, lifetime, 'beam_chunk')
                particles.append(particle)
        
        # Create fewer fire/explosion particles for faster animation (faster than beam chunks)
        for i in range(intensity // 2):  # Half the particles for better performance
            # Random angle and higher speed for sparks (faster than beam chunks)
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(400, 800)  # Much faster speed range for sparks
            velocity = np.array([np.cos(angle) * speed, np.sin(angle) * speed])
            # Random colors (fire-like: red, orange, yellow)
            color_choices = [
                (255, 100, 50),   # Orange-red
                (255, 150, 50),   # Orange
                (255, 200, 50),   # Yellow-orange
                (255, 80, 80),    # Red
                (255, 255, 100),  # Yellow
                (200, 200, 200),  # White/gray smoke
            ]
            color = color_choices[np.random.randint(0, len(color_choices))]
            # Random size and very short lifetime for much faster animation
            size = np.random.uniform(3, 8)  # Slightly smaller sparks
            lifetime = np.random.uniform(0.2, 0.7)  # Much shorter lifetime for faster animation
            particle = ExplosionParticle(center_pos, velocity, color, size, lifetime, 'circle')
            particles.append(particle)
            
        self.explosions.append({
            'particles': particles,
            'age': 0.0,
            'max_lifetime': 1.3,  # Much shorter total explosion duration
            'center_pos': center_pos,  # Store center position for flash effect
            'flash_duration': 0.2,  # Duration of initial flash in seconds
            'flash_intensity': 255,  # Maximum flash brightness
            'flash_radius': 80  # Radius of flash effect
        })
        
    def update(self, dt):
        """Update all active explosions"""
        active_explosions = []
        
        for explosion in self.explosions:
            explosion['age'] += dt
            
            # Update all particles in this explosion
            active_particles = []
            for particle in explosion['particles']:
                if particle.update(dt):
                    active_particles.append(particle)
            
            explosion['particles'] = active_particles
            
            # Keep explosion if it has particles or hasn't reached max lifetime
            if explosion['particles'] or explosion['age'] < explosion['max_lifetime']:
                active_explosions.append(explosion)
                
        self.explosions = active_explosions
        
    def draw(self, surf):
        """Draw all active explosions with initial flash effect"""
        for explosion in self.explosions:
            # Draw initial flash effect
            if explosion['age'] < explosion['flash_duration']:
                # Calculate flash alpha based on age (fade out quickly)
                flash_progress = explosion['age'] / explosion['flash_duration']
                flash_alpha = int(explosion['flash_intensity'] * (1.0 - flash_progress))
                
                if flash_alpha > 0:
                    # Create flash surface
                    flash_radius = int(explosion['flash_radius'])
                    flash_surface = pygame.Surface((flash_radius * 2, flash_radius * 2), pygame.SRCALPHA)
                    
                    # Draw bright white flash circle
                    flash_color = (255, 70, 0, flash_alpha)
                    pygame.draw.circle(flash_surface, flash_color, (flash_radius, flash_radius), flash_radius)
                    
                    # Draw smaller, brighter inner flash
                    inner_radius = int(flash_radius * 0.6)
                    inner_alpha = min(255, int(flash_alpha * 1.5))
                    inner_color = (255, 255, 30, inner_alpha)  # Slightly yellow-white
                    pygame.draw.circle(flash_surface, inner_color, (flash_radius, flash_radius), inner_radius)
                    
                    # Position flash at explosion center
                    flash_rect = flash_surface.get_rect(center=(int(explosion['center_pos'][0]), int(explosion['center_pos'][1])))
                    surf.blit(flash_surface, flash_rect)
            
            # Draw particles
            for particle in explosion['particles']:
                particle.draw(surf)
                
    def has_active_explosions(self):
        """Check if there are any active explosions"""
        return len(self.explosions) > 0

# Global instances
geometry_cache = GeometryCache()
mouse_cache = MouseCache()
explosion_system = ExplosionSystem()

def draw_disclaimer(surf):
    """Draws a small Creative Commons license disclaimer in the lower right corner."""
    # Two-line, smaller, short-name Creative Commons disclaimer in lower right
    lines = [
        "© 2025 S.Schanz",
        "CC BY-NC 4.0"
    ]
    font = get_font('values')  # Use smaller font
    surf_rect = surf.get_rect()
    # Render both lines
    disclaimer1 = font.render(lines[0], True, COLORS['c2025'])
    disclaimer2 = font.render(lines[1], True, COLORS['c2025'])
    # Position: bottom right, stacked, right-aligned
    total_height = disclaimer1.get_height() + disclaimer2.get_height()
    y = surf_rect.bottom - total_height - EDGE_MARGIN
    x1 = surf_rect.right - disclaimer1.get_width() - EDGE_MARGIN
    x2 = surf_rect.right - disclaimer2.get_width() - EDGE_MARGIN
    surf.blit(disclaimer1, (x1, y))
    surf.blit(disclaimer2, (x2, y + disclaimer1.get_height()))

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
        self._support_reactions_cache = None  # Cache for expensive calculations

    def global_to_local(self, v):
        return np.array([np.dot(v, self.e_x), np.dot(v, self.e_z)])

    def _invalidate_cache(self):
        """Invalidate cached calculations when beam state changes"""
        self._support_reactions_cache = None

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
        self._invalidate_cache()

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
        self._invalidate_cache()

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
        
        self._invalidate_cache()
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

    def get_center_of_gravity(self):
        """Calculate the center of gravity (geometric center) of the beam"""
        # For a simple beam, the center of gravity is at the midpoint
        return (self.start + self.end) / 2

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
            # Always assign q1 to the amplitude at the lower x, q2 to the higher x
            if x1 <= x2:
                x_low, x_high = x1, x2
                q_low, q_high = start_amplitude, end_amplitude
            else:
                x_low, x_high = x2, x1
                q_low, q_high = end_amplitude, start_amplitude
            length = x_high - x_low
            if length > 0:
                # Only consider active area to the right of the cut
                if x_high <= x:
                    continue  # Complete load left of or at the cut
                x_left = max(x, x_low)  # From cut or load start
                x_right = x_high        # To load end
                if x_left < x_right:
                    # Active segment length and local coordinates (in pixels)
                    l_active = x_right - x_left
                    xi1 = x_left - x_low  # Local coordinate at cut (pixels)
                    xi2 = x_right - x_low # Local coordinate at end (pixels)
                    L_m = length / GRID_SIZE
                    xi1_m = xi1 / GRID_SIZE
                    xi2_m = xi2 / GRID_SIZE
                    q1 = q_low
                    q2 = q_high
                    # Resultant force of linear load in active region (N)
                    F_res = q1 * (xi2_m - xi1_m) + (q2 - q1) * ((xi2_m**2 - xi1_m**2) / (2 * L_m))
                    # Center of gravity of linear load in active region (in meters from beam start)
                    if abs(F_res) > 1e-12:
                        moment_integral = q1 * (xi2_m**2 - xi1_m**2) / 2 + (q2 - q1) * (xi2_m**3 - xi1_m**3) / (3 * L_m)
                        x_centroid_local_m = moment_integral / F_res + x_low / GRID_SIZE
                    else:
                        x_centroid_local_m = (x_left + x_right) / (2 * GRID_SIZE)
                    # Contributions to internal forces
                    Q += F_res
                    distance_m = x_centroid_local_m - (x / GRID_SIZE)
                    M -= F_res * distance_m
        
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
        Uses caching for performance optimization.
        """
        # Return cached result if available
        if self._support_reactions_cache is not None:
            return self._support_reactions_cache
            
        # First check static determinacy
        is_determinate, _ = self.check_static_determinacy()
        if not is_determinate:
            self._support_reactions_cache = {}
            return self._support_reactions_cache
            
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
            # Always assign q1 to the amplitude at the lower x, q2 to the higher x
            if x1 <= x2:
                x_low, x_high = x1, x2
                q_low, q_high = start_amplitude, end_amplitude
            else:
                x_low, x_high = x2, x1
                q_low, q_high = end_amplitude, start_amplitude
            length = x_high - x_low
            if length > 0:
                q1 = q_low
                q2 = q_high
                L_m = length / GRID_SIZE
                F_res = (q1 + q2) * L_m / 2.0
                sum_Fx += 0
                sum_Fz += F_res
                # Robust centroid for all cases (measured from x_low)
                if abs(q1) < 1e-12 and abs(q2) > 1e-12:
                    # Triangle: left zero, right q2
                    x_centroid = x_low + (2/3) * length
                elif abs(q2) < 1e-12 and abs(q1) > 1e-12:
                    # Triangle: right zero, left q1
                    x_centroid = x_low + (1/3) * length
                elif abs(q1 + q2) > 1e-12:
                    # General trapezoid
                    x_centroid = x_low + length * (2 * q2 + q1) / (3 * (q1 + q2))
                else:
                    x_centroid = x_low + length / 2

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
                            distance_from_end_m = (self.L - x_l)  # Convert pixels to meters
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
                                distance_from_end_m = (self.L - x_centroid)  # Convert pixels to meters
                                m_about_end += F_res * distance_from_end_m  # Distance from end in meters
                        
                        m_end = -m_about_end  # Reaction moment
                        support_reactions["end"] = (fx_end, fz_end, m_end)
                        
                        # First support redundant/overdetermined
                        support_reactions["start"] = (0, 0, 0)
        
        # Cache the result for performance
        self._support_reactions_cache = support_reactions
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
            text = font_values.render(f"{force_value:.0f} N", True, text_color)
            
            # Position text in force direction with distance from arrow tip
            if force_norm > 0:
                force_unit = force_global / force_norm
                text_offset = force_unit * MIN_ARROW_SPACING
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
                # Show actual applied load intensity in N/m
                n_per_m = abs(end_amplitude)
                text = font_values.render(f"{n_per_m:.0f} N/m", True, text_color)

                # Position text in the middle
                mid_pos = (start_pos + end_pos) / 2
                if abs(end_amplitude) > 0:
                    force_unit = force_vector_end / np.linalg.norm(force_vector_end)
                    text_offset = force_unit * MIN_ARROW_SPACING
                    text_pos = mid_pos + force_vector_end + text_offset
                else:
                    text_pos = mid_pos + np.array([5, -5])
                # Center the text at the position like preview does
                text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text, text_rect)
            else:
                # Variable load - always show both values (including 0 N/m)
                start_n_per_m = abs(start_amplitude)
                end_n_per_m = abs(end_amplitude)

                # Start value - always show, including 0 N/m
                text_start = font_values.render(f"{start_n_per_m:.0f} N/m", True, text_color)
                if abs(start_amplitude) > 0:
                    force_unit = force_vector_start / np.linalg.norm(force_vector_start)
                    text_offset = force_unit * MIN_ARROW_SPACING
                    text_pos = start_pos + force_vector_start + text_offset
                else:
                    text_pos = start_pos + np.array([5, -15])  # Position for zero values
                # Center the text at the position like preview does
                text_rect = text_start.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text_start, text_rect)

                # End value - always show, including 0 N/m
                text_end = font_values.render(f"{end_n_per_m:.0f} N/m", True, text_color)
                if abs(end_amplitude) > 0:
                    force_unit = force_vector_end / np.linalg.norm(force_vector_end)
                    text_offset = force_unit * MIN_ARROW_SPACING
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
                    force_vector = self.e_z * fz * 0.5 * scale_factor
                    tip = pos + force_vector
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, FORCE_LINE_THICKNESS)

                    # Optimized arrow head for z-direction
                    force_unit = force_vector / np.linalg.norm(force_vector)
                    triangle_width = 6 * ARROW_SIZE_RATIO  # Smaller for reaction forces
                    arrow_points = geometry_cache.get_arrow_points(tip, force_unit, 6, triangle_width)
                    pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)

                    # Text rendering (no rounding)
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fz:.1f} N", True, COLORS['reaction'])
                    text_offset = force_unit * MIN_ARROW_SPACING
                    text_pos = tip + text_offset
                    text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                    surf.blit(text, text_rect)

                # Reaction force in x-direction (optimized, now scaled with scale_factor)
                if abs(fx) > 0.1:
                    force_vector = self.e_x * fx * 0.5 * scale_factor
                    tip = pos + force_vector
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, FORCE_LINE_THICKNESS)

                    # Optimized arrow head for x-direction
                    force_unit = force_vector / np.linalg.norm(force_vector)
                    triangle_width = 6 * ARROW_SIZE_RATIO
                    arrow_points = geometry_cache.get_arrow_points(tip, force_unit, 6, triangle_width)
                    pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)

                    # Text rendering (no rounding)
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fx:.1f} N", True, COLORS['reaction'])
                    surf.blit(text, (tip + np.array([5, 5])).astype(int))
                    
        # Draw supports last so they overlay everything (including graphs)
        for support_pos, support_type in self.supports.items():
            is_highlighted = highlight_item == ('support', support_pos)
            if support_pos == "start":
                self.draw_support(surf, self.start, support_type, is_highlighted)
            elif support_pos == "end":
                self.draw_support(surf, self.end, support_type, is_highlighted)

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
                pts_N.append(w - self.e_z * N * scale_factor)
                pts_Q.append(w - self.e_z * Q * scale_factor)
                # Use smaller scaling for moment display (visual only, actual values remain correct)
                pts_M.append(w - self.e_z * M * scale_factor * 0.1)
        
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

    def draw_significant_values(self, surf, segments, scale_factor):
        """Draw segmentwise significant values (start/end points) on internal force diagrams"""
        if not segments:
            return
        # Suppress all annotations if no loads are applied
        if not self.point_loads and not self.line_loads:
            return
        font_values = get_font('values')
        significant_points = []
        # Precompute if each force type is zero everywhere
        force_types = ['N', 'Q', 'M']
        zero_everywhere = {}
        for force_type in force_types:
            idx = force_types.index(force_type)
            values = [self.internal_forces(x)[idx] for x in segments]
            zero_everywhere[force_type] = all(abs(v) < 1e-10 for v in values)
        # Always show start and end points for each force type
        for force_type in force_types:
            x_start = segments[0]
            value_start = self.internal_forces(x_start)[force_types.index(force_type)]
            significant_points.append((x_start, value_start, force_type, 'segment_start'))
            x_end = segments[-1]
            value_end = self.internal_forces(x_end)[force_types.index(force_type)]
            significant_points.append((x_end, value_end, force_type, 'segment_end'))
        for i in range(len(segments) - 1):
            x0 = segments[i]
            x1 = segments[i + 1]
            for force_type in force_types:
                v0 = self.internal_forces(x0)[force_types.index(force_type)]
                v1 = self.internal_forces(x1)[force_types.index(force_type)]
                significant_points.append((x0, v0, force_type, 'segment_start'))
                significant_points.append((x1, v1, force_type, 'segment_end'))
        # Remove duplicates (by x, value, force_type)
        unique_points = []
        seen = set()
        for pt in significant_points:
            key = (round(pt[0], 6), round(pt[1], 2), pt[2])
            if key not in seen:
                unique_points.append(pt)
                seen.add(key)
        # Only show one zero annotation per position (no overlap for N, Q, M)
        zero_positions = set()
        zero_annotated = set()
        for x, value, force_type, point_type in unique_points:
            is_zero = abs(value) < 1e-10
            pos_key = (round(x, 2))
            # Suppress zero annotation if graph is zero everywhere for this force type
            if is_zero and zero_everywhere.get(force_type, False):
                continue
            if is_zero:
                # Only annotate if value is exactly zero
                if value != 0.0:
                    continue
                if pos_key in zero_positions:
                    continue
                zero_positions.add(pos_key)
                zero_annotated.add(force_type)
                color = (180, 180, 180)  # Grey color for zero annotation
            else:
                if force_type == 'N':
                    color = COLORS['N']
                elif force_type == 'Q':
                    color = COLORS['Q']
                else:
                    color = COLORS['M']
            # Only annotate zero once per position, even if multiple force types
            if is_zero and list(zero_annotated).count(force_type) > 1:
                continue
            w = self.world_point(x)
            if force_type == 'N':
                graph_pos = w - self.e_z * value * scale_factor
            elif force_type == 'Q':
                graph_pos = w - self.e_z * value * scale_factor
            else:
                graph_pos = w - self.e_z * value * scale_factor * 0.1
            if is_zero:
                value_text = "0.0"
            elif force_type == 'M':
                value_text = f"{value:.1f} Nm"
            else:
                value_text = f"{value:.1f} N"
            text_surface = font_values.render(value_text, True, color)
            text_offset = self.e_z * (MIN_ARROW_SPACING if value >= 0 else -MIN_ARROW_SPACING)
            text_pos = graph_pos - text_offset
            text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            pygame.draw.circle(surf, color, graph_pos.astype(int), 4)
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
            priority = {
                'zero': 1, 'discontinuity': 2, 'maximum': 3, 'minimum': 3, 'constant': 4
            }
            must_keep = []
            others = []
            for point in type_points:
                x, value, force_type, point_type = point
                if point_type in ['zero', 'discontinuity'] or abs(value) > 10:
                    must_keep.append(point)
                else:
                    others.append(point)
            # Special handling for constant segments: only keep start and end
            constant_points = [p for p in type_points if p[3] == 'constant']
            if len(constant_points) > 2:
                constant_points = [constant_points[0], constant_points[-1]]
            # Add must-keep points and constant segment endpoints
            filtered_points.extend(must_keep)
            filtered_points.extend(constant_points)
            # Filter others by distance
            last_x = -float('inf')
            for point in others:
                x, value, force_type, point_type = point
                if x - last_x >= min_distance:
                    filtered_points.append(point)
                    last_x = x
            # Limit total points per force type
            type_filtered = [p for p in filtered_points if p[2] == force_type]
            if len(type_filtered) > 6:
                type_filtered.sort(key=lambda p: (priority.get(p[3], 5), -abs(p[1])))
                filtered_points = [p for p in filtered_points if p[2] != force_type] + type_filtered[:6]
        
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

def draw_slider(surf, value, min_val, max_val, width, label):
    """Draws a slider control, positioned EDGE_MARGIN from the upper right corner."""
    SLIDER_THICKNESS = 20

    # Calculate x so the slider is EDGE_MARGIN from the right edge
    surf_rect = surf.get_rect()
    slider_x = surf_rect.right - width - EDGE_MARGIN
    slider_y = EDGE_MARGIN


    # Slider body
    pygame.draw.rect(surf, COLORS['slider_bg_color'], (slider_x, slider_y, width, SLIDER_THICKNESS), 0, border_radius=SLIDER_THICKNESS // 2)

    # Draw vertical mark at zero scale (middle of slider)
    zero_pos = slider_x + width // 2
    pygame.draw.line(surf, COLORS['slider_mark_color'], (zero_pos, slider_y +4), (zero_pos, slider_y + SLIDER_THICKNESS -5), 2)

    # Calculate slider handle position so knob stays within rounded corners
    slider_pos = slider_x + (SLIDER_THICKNESS // 2) + (value - min_val) / (max_val - min_val) * (width - SLIDER_THICKNESS)

    # Dimmer blue slider handle
    pygame.draw.circle(surf, COLORS['ui_text'], (int(slider_pos), slider_y + SLIDER_THICKNESS // 2), SLIDER_THICKNESS // 2 -2)

    # Position label and value centered under the slider
    font_slider = get_font('slider')
    # Always use COLORS['ui_text'] for the slider label and value, regardless of state
    label_text = font_slider.render(f"{label}: {value:.1f}", True, COLORS['ui_text'])
    text_rect = label_text.get_rect()
    text_x = slider_x + (width - text_rect.width) // 2
    surf.blit(label_text, (text_x, slider_y + SLIDER_THICKNESS + 5))  # 5px under the slider

    return (slider_x, slider_y, width, SLIDER_THICKNESS)  # For collision detection

def handle_slider_click(mouse_pos, slider_rect, min_val, max_val):
    """Handles clicks on the slider with extended click area"""
    x, y, width, height = slider_rect
    # Extended click area: 10 pixels above and below the slider
    extended_y = y - CLICK_PROXIMITY
    extended_height = height + 2 * CLICK_PROXIMITY

    knob_radius = SLIDER_THICKNESS / 2
    knob_min = x + knob_radius
    knob_max = x + width - knob_radius
    # Allow clicks within the slider track (excluding rounded corners)
    if knob_min <= mouse_pos[0] <= knob_max and extended_y <= mouse_pos[1] <= extended_y + extended_height:
        # Set value so that the knob center is exactly under the mouse x
        rel = (mouse_pos[0] - knob_min) / (width - SLIDER_THICKNESS)
        value = min_val + rel * (max_val - min_val)
        return max(min_val, min(max_val, value))
    return None

def find_item_under_mouse(mouse_pos, beam, detection_radius=15):
    """Find which item (point load, line load, support, or beam) is under the mouse cursor"""
    if not beam:
        return None, None
    
    # Check point loads first (highest priority)
    for i, (pos_global, force_global) in enumerate(beam.point_loads):
        # Check proximity to the point load start position using cached distance
        if geometry_cache.get_distance(mouse_pos, pos_global) <= detection_radius:
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
            
            # Check if mouse is close to the arrow line using cached distance
            if geometry_cache.get_distance(mouse_pos, closest_point) <= detection_radius:
                return ('point_load', i), pos_global
    
    # Check line loads (check both start and end positions, and the polygon area)
    for i, (start_pos, end_pos, end_amplitude, start_amplitude) in enumerate(beam.line_loads):
        # Check start position using cached distance
        if geometry_cache.get_distance(mouse_pos, start_pos) <= detection_radius:
            return ('line_load', i), start_pos
        # Check end position using cached distance
        if geometry_cache.get_distance(mouse_pos, end_pos) <= detection_radius:
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
        beam._invalidate_cache()
        return beam
    elif item_category == 'line_load' and 0 <= index_or_pos < len(beam.line_loads):
        del beam.line_loads[index_or_pos]
        beam._invalidate_cache()
        return beam
    elif item_category == 'support' and index_or_pos in beam.supports:
        del beam.supports[index_or_pos]
        beam._invalidate_cache()
        return beam
    elif item_category == 'beam':
        # Delete the entire beam (return None to indicate beam should be deleted)
        return None
    
    return beam

def draw_ui(screen, mode, beam, scale_factor, clicks):
    """Draw the user interface elements"""
    # Draw scale slider only when statically determinate
    slider_rect = None
    if beam:
        is_determinate, _ = beam.check_static_determinacy()
        if is_determinate:
            # Position slider in upper right corner - same size, wider range
            slider_rect = draw_slider(screen, scale_factor, SLIDER_MIN, SLIDER_MAX, SLIDER_WIDTH, "Graph Scale")

    # Shortcuts display
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
        x_offset = EDGE_MARGIN
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
    render_highlighted_shortcuts(shortcuts_line1, EDGE_MARGIN)
    render_highlighted_shortcuts(shortcuts_line2, EDGE_MARGIN + font_ui.get_height() + 5)
    # Show static determinacy bottom left above system dialogs
    if beam:
        is_determinate, status_text = beam.check_static_determinacy()
        status_color = COLORS['ui_text'] if is_determinate else COLORS['status_error']
        status_display = font_ui.render(status_text, True, status_color)
        screen.blit(status_display, (EDGE_MARGIN, screen.get_height() - EDGE_MARGIN - 2 * status_display.get_height() - 5))  # 25 pixels above system dialogs
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
        screen.blit(status_text, (EDGE_MARGIN, screen.get_height() - status_text.get_height() - EDGE_MARGIN))

    return slider_rect

# Main game loop variables
geometry_cache = GeometryCache()
beam = None
mode = "idle"
clicks = []
temp_beam = None  # Temporary beam for preview
scale_factor = 1.0  # Scaling factor for internal force diagrams  
slider_dragging = False
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
        wave_phase = t * wave_params['wave_cycles'] * 2 * np.pi
        wave_offset = wave_params['perpendicular'] * wave_amplitude * np.sin(
            2 * np.pi * wave_frequency * animation_time + wave_phase + phase_offset
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
        wave_phase = edge_t * wave_params['wave_cycles'] * 2 * np.pi
        wave_offset = wave_params['perpendicular'] * wave_amplitude * np.sin(
            2 * np.pi * wave_frequency * animation_time + wave_phase + phase_offset
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
    end_wave_phase = wave_params['wave_cycles'] * 2 * np.pi
    wave_connection_offset = wave_params['perpendicular'] * wave_amplitude * np.sin(
        2 * np.pi * wave_frequency * animation_time + end_wave_phase + phase_offset
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

    slider_rect = draw_ui(screen, mode, beam, scale_factor, clicks)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = mouse_cache.get_snapped_mouse_pos()

            # Check slider interaction first only if slider_rect is available
            if slider_rect is not None:
                new_scale = handle_slider_click(mouse_cache.get_mouse_pos(), slider_rect, SLIDER_MIN, SLIDER_MAX)
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
                        # Beam was deleted - create explosion at mouse position
                        explosion_center = mouse_cache.get_snapped_mouse_pos()  # Use mouse position
                        beam_color = COLORS['beam']  # Get the purple beam color
                        explosion_system.create_explosion(explosion_center, intensity=40, beam_color=beam_color)
                        
                        # Reset everything
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
            # Update slider during dragging only if slider_rect is available
            if slider_rect is not None:
                new_scale = handle_slider_click(mouse_cache.get_mouse_pos(), slider_rect, SLIDER_MIN, SLIDER_MAX)
                if new_scale is not None:
                    scale_factor = new_scale

        elif event.type == pygame.MOUSEMOTION:
            # Handle delete mode highlighting
            if mode == "delete" and beam:
                mouse_pos = mouse_cache.get_snapped_mouse_pos()
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

    # Vorschau während der Erstellung
    mpos = mouse_cache.get_snapped_mouse_pos()
    
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
            length_text = font_preview.render(f"{beam_length_meters:.1f} m", True, COLORS['beam'])  # Match beam color
            
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
            end_wave_phase = wave_params['wave_cycles'] * 2 * np.pi
            wave_connection_offset = wave_params['perpendicular'] * ANIMATION_AMPLITUDE * np.sin(
                2 * np.pi * ANIMATION_FREQUENCY * animation_time + end_wave_phase
            )
            animated_wave_end = wave_end_point + wave_connection_offset
            animated_tip = tip + wave_connection_offset
            
            # Draw connecting line and arrowhead
            pygame.draw.line(screen, COLORS['force_preview'], animated_wave_end, animated_tip, FORCE_LINE_THICKNESS)
            
            # Animated arrowhead using geometry cache
            force_unit = wave_params['force_unit']
            triangle_width = ARROW_HEAD_SIZE * ARROW_SIZE_RATIO  # Use variables for triangle base
            
            # Use geometry cache for arrow points (animated tip position)
            arrow_points = geometry_cache.get_arrow_points(
                animated_tip, force_unit, ARROW_HEAD_SIZE, triangle_width
            )
            pygame.draw.polygon(screen, COLORS['force_preview'], arrow_points)
            
            # Display load intensity - STATIC position
            load_intensity = f"{force_norm:.0f} N"
            font_preview = get_font('preview')
            text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
            text_offset = force_unit * MIN_ARROW_SPACING
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
            
            # Display load intensity (show actual amplitude in N/m)
            if num_arrows > 0 and np.linalg.norm(force_vector) > 5:
                amplitude = np.dot(mpos - 0.5 * (clicks[0] + clicks[1]), beam.e_z)
                load_intensity = f"{abs(amplitude):.0f} N/m"
                font_preview = get_font('preview')
                text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
                force_norm = np.linalg.norm(force_vector)
                force_unit = force_vector / force_norm
                mid_point = (clicks[0] + clicks[1]) / 2
                arrow_end = mid_point + force_vector
                text_offset = force_unit * MIN_ARROW_SPACING
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
        # Trapezoidal load direction preview (define one side, show value above that side)
        if beam:
            mid = 0.5 * (clicks[0] + clicks[1])
            mouse_vec = mpos - mid
            amplitude = np.dot(mouse_vec, beam.e_z)
            force_vector = beam.e_z * amplitude
            
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            
            # If the line load is shorter than minimum spacing, only show edge arrows
            if line_length < LINE_LOAD_SPACING * 2:
                num_arrows = 2
            else:
                num_arrows = int(line_length / LINE_LOAD_SPACING) + 1
                num_arrows = max(2, num_arrows)
            
            wave_params = calculate_wave_parameters(force_vector, ANIMATION_ARROW_HEAD_LENGTH, ANIMATION_PERIOD_LENGTH)
            
            if wave_params:
                animated_polygon_points = create_animated_polygon(
                    clicks, wave_params, wave_params, ANIMATION_SEGMENTS, animation_time,
                    ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, force_vector, force_vector
                )
                draw_transparent_polygon(screen, COLORS['force_preview'], animated_polygon_points, 180)
            else:
                rect_points = [clicks[0], clicks[1], clicks[1] + force_vector, clicks[0] + force_vector]
                draw_transparent_polygon(screen, COLORS['force_preview'], rect_points, 180)
            
            for i in range(num_arrows):
                t = i / (num_arrows - 1) if num_arrows > 1 else 0
                arrow_start = clicks[0] + t * (clicks[1] - clicks[0])
                phase_offset = i * 0.3
                draw_animated_arrow(screen, arrow_start, force_vector, wave_params, phase_offset,
                                  animation_time, ANIMATION_FREQUENCY, ANIMATION_AMPLITUDE, 
                                  ANIMATION_SEGMENTS, ANIMATION_ARROW_HEAD_LENGTH, COLORS['force_preview'])
            
            # Show the preview value above the side being defined (let's use the END side for clarity)
            if num_arrows > 0 and np.linalg.norm(force_vector) > 5:
                load_intensity = f"{abs(amplitude):.0f} N/m"
                font_preview = get_font('preview')
                text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
                force_norm = np.linalg.norm(force_vector)
                force_unit = force_vector / force_norm if force_norm > 0 else np.array([0, -1])
                # Show value above the END side (clicks[1])
                text_offset = force_unit * MIN_ARROW_SPACING
                text_pos = clicks[1] + force_vector + text_offset
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
            
            # Display both start and end intensities at their respective positions
            font_preview = get_font('preview')
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            if line_length > 0:
                # Start intensity (show actual amplitude in N/m) at start position
                start_text = f"{abs(start_amplitude):.0f} N/m"
                text_surface_start = font_preview.render(start_text, True, COLORS['force_text'])
                force_unit_start = force_vector_start / np.linalg.norm(force_vector_start) if np.linalg.norm(force_vector_start) > 0 else np.array([0, -1])
                text_offset_start = force_unit_start * MIN_ARROW_SPACING
                text_pos_start = clicks[0] + force_vector_start + text_offset_start
                text_rect_start = text_surface_start.get_rect(center=(int(text_pos_start[0]), int(text_pos_start[1])))
                screen.blit(text_surface_start, text_rect_start)

                # End intensity (show actual amplitude in N/m) at end position
                end_text = f"{abs(end_amplitude):.0f} N/m"
                text_surface_end = font_preview.render(end_text, True, COLORS['force_text'])
                force_unit_end = force_vector_end / np.linalg.norm(force_vector_end) if np.linalg.norm(force_vector_end) > 0 else np.array([0, -1])
                text_offset_end = force_unit_end * MIN_ARROW_SPACING
                text_pos_end = clicks[1] + force_vector_end + text_offset_end
                text_rect_end = text_surface_end.get_rect(center=(int(text_pos_end[0]), int(text_pos_end[1])))
                screen.blit(text_surface_end, text_rect_end)

    # Update and draw explosion system
    dt = clock.get_time() / 1000.0  # Delta time in seconds
    explosion_system.update(dt)
    explosion_system.draw(screen)

    # Update animation time for oscillating preview effects
    animation_time += dt  # Use the calculated dt

    draw_disclaimer(screen)
    pygame.display.flip()
    clock.tick(120)

pygame.quit()
sys.exit()