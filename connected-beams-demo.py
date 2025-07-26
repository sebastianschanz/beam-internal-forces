import pygame
import numpy as np
import sys

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Connected Beams - Full Structural Analysis")
clock = pygame.time.Clock()

# Color System - Same as original
COLORS = {
    # Background and Grid
    'bg': (30, 30, 40),
    'grid': (60, 60, 70),
    
    # UI Elements
    'ui_text': (120, 180, 250),
    'ui_active': (255, 255, 255),
    'ui_bg': (80, 80, 90),
    'status_ok': (120, 180, 250),
    'status_error': (255, 80, 80),
    
    # Structural Elements
    'beam': (170, 100, 190),
    'symbol_bg': (55, 55, 90),
    'symbol_line': (170, 100, 190),
    
    # Forces and Analysis
    'force_preview': (210, 20, 80),
    'force_display': (210, 20, 80),
    'force_line': (210, 20, 80),
    'force_text': (120, 150, 200),
    'reaction': (40, 40, 220),
    'reaction_text': (120, 150, 200),
    
    # Coordinate System
    'x_axis': (200, 40, 40),
    'z_axis': (30, 160, 30),
    
    # Internal Forces (N, Q, M)
    'N': (200, 150, 30),      # Normal force - red
    'Q': (100, 200, 100),    # Shear force - green  
    'M': (80, 140, 220),     # Moment - blue
    
    # Joint types
    'joint_rigid': (255, 200, 100),      # Rigid connection with visible joint
    'joint_pin': (100, 255, 100),        # Pin connection
    'joint_free': (255, 100, 100),       # No connection
    'rigid_connection': (170, 100, 190), # Rigid connection without visible joint (same as beam color)
    'selected': (255, 255, 100),
}

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

# Joint types
JOINT_RIGID = 0       # Full continuity (M and Q transfer) - visible joint symbol
JOINT_PIN = 1         # Moment release (only Q transfers) - visible joint symbol  
JOINT_FREE = 2        # No connection (independent beams) - visible joint symbol
RIGID_CONNECTION = 3  # Full continuity but no visible joint (continuous polygon)

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

def get_font(font_key):
    """Get font by key"""
    if isinstance(font_key, str):
        family, size = FONTS[font_key]
        return pygame.font.SysFont(family, size)
    else:
        # Fallback for size-only calls
        return pygame.font.SysFont('consolas', font_key)

class ConnectedBeamStructure:
    """Manages multiple connected beams as a unified structural system"""
    def __init__(self):
        self.beams = []
        self.joints = {}  # {position_key: {'beams': [beam_indices], 'type': JOINT_TYPE}}
        self.joint_tolerance = 8.0
        
    def add_beam(self, start, end):
        """Add a beam and update joint connections"""
        # Check if this beam connects to existing structure (unless it's the first beam)
        if len(self.beams) > 0:
            connects_to_existing = False
            for existing_beam in self.beams:
                # Check if new beam shares a point with any existing beam
                start_dist_to_existing_start = np.linalg.norm(start - existing_beam.start)
                start_dist_to_existing_end = np.linalg.norm(start - existing_beam.end)
                end_dist_to_existing_start = np.linalg.norm(end - existing_beam.start)
                end_dist_to_existing_end = np.linalg.norm(end - existing_beam.end)
                
                if (start_dist_to_existing_start < self.joint_tolerance or 
                    start_dist_to_existing_end < self.joint_tolerance or
                    end_dist_to_existing_start < self.joint_tolerance or 
                    end_dist_to_existing_end < self.joint_tolerance):
                    connects_to_existing = True
                    break
            
            # Don't add beam if it doesn't connect to existing structure
            if not connects_to_existing:
                return None  # Beam rejected - not connected to structure
        
        beam = Beam(start, end, len(self.beams))
        beam.connected_structure = self  # Link beam to structure
        self.beams.append(beam)
        self._update_joints(beam)
        
        # Reorder beams from left to right and update beam IDs
        self._reorder_beams_left_to_right()
        
        return beam
    
    def _reorder_beams_left_to_right(self):
        """Reorder beams from left to right and update their IDs and coordinate systems"""
        if len(self.beams) <= 1:
            return
            
        # Sort beams by their leftmost x-coordinate
        def get_leftmost_x(beam):
            return min(beam.start[0], beam.end[0])
        
        # Sort beams by leftmost position
        sorted_beams = sorted(self.beams, key=get_leftmost_x)
        
        # Update beam IDs and ensure coordinate systems flow left to right
        for i, beam in enumerate(sorted_beams):
            beam.beam_id = i
            
            # Ensure local coordinate system flows left to right
            if beam.start[0] > beam.end[0]:  # If start is to the right of end
                # Swap start and end to make coordinate system flow left to right
                beam.start, beam.end = beam.end, beam.start
                
            # Recalculate unit vectors after potential swap
            beam.L = np.linalg.norm(beam.end - beam.start)
            if beam.L > 0:
                beam.e_x = (beam.end - beam.start) / beam.L  # Unit vector along beam
                beam.e_z = np.array([-beam.e_x[1], beam.e_x[0]])  # Perpendicular unit vector
        
        # Update the beams list with the reordered beams
        self.beams = sorted_beams
        
        # Update joint references to reflect new beam indices
        self._update_joint_beam_indices()
    
    def _update_joint_beam_indices(self):
        """Update joint beam indices after reordering beams"""
        # Create a mapping from old beam objects to new indices
        beam_to_new_id = {beam: beam.beam_id for beam in self.beams}
        
        # Clear and rebuild joints with correct beam indices
        new_joints = {}
        for joint_key, joint_data in self.joints.items():
            joint_pos = np.array(joint_key)
            new_beam_indices = []
            
            # Find which beams are actually at this joint position
            for beam in self.beams:
                start_dist = np.linalg.norm(beam.start - joint_pos)
                end_dist = np.linalg.norm(beam.end - joint_pos)
                
                if start_dist < self.joint_tolerance or end_dist < self.joint_tolerance:
                    new_beam_indices.append(beam.beam_id)
            
            if len(new_beam_indices) > 0:
                new_joints[joint_key] = {
                    'beams': new_beam_indices,
                    'type': joint_data.get('type', RIGID_CONNECTION)
                }
        
        self.joints = new_joints

    def _update_joints(self, new_beam):
        """Update joint connections with default rigid joints"""
        beam_idx = len(self.beams) - 1
        
        for point in [new_beam.start, new_beam.end]:
            joint_key = self._find_or_create_joint(point)
            joint_data = self.joints[joint_key]
            
            if beam_idx not in joint_data['beams']:
                joint_data['beams'].append(beam_idx)
                
                # Set default joint type based on number of beams
                if len(joint_data['beams']) > 1:
                    joint_data['type'] = RIGID_CONNECTION  # Default to seamless rigid connection
    
    def _find_or_create_joint(self, point):
        """Find existing joint near point or create new one"""
        # Check for nearby existing joints
        for joint_key, joint_data in self.joints.items():
            joint_pos = np.array(joint_key)
            if np.linalg.norm(point - joint_pos) < self.joint_tolerance:
                return joint_key
        
        # Create new joint with default rigid connection type (seamless)
        key = (round(point[0]), round(point[1]))
        self.joints[key] = {'beams': [], 'type': RIGID_CONNECTION}
        return key
    
    def toggle_joint_type(self, point):
        """Toggle joint type at given position"""
        for joint_key, joint_data in self.joints.items():
            joint_pos = np.array(joint_key)
            if np.linalg.norm(point - joint_pos) < self.joint_tolerance:
                if len(joint_data['beams']) > 1:  # Only toggle if multiple beams
                    # Cycle through joint types: RIGID -> PIN -> FREE -> RIGID_CONNECTION -> RIGID
                    joint_data['type'] = (joint_data['type'] + 1) % 4
                return joint_data.get('type', JOINT_FREE)
        return None
    
    def check_overall_static_determinacy(self):
        """Check static determinacy for the entire connected structure"""
        if len(self.beams) == 0:
            return False, "No beams in structure"
            
        # Count total degrees of freedom and constraints
        n = len(self.beams)  # Number of beams
        j = len([joint for joint in self.joints.values() if len(joint.get('beams', [])) > 1])  # Connected joints
        
        # Count support reactions
        s = 0
        for beam in self.beams:
            for lager_pos, lager_typ in beam.lager.items():
                if lager_typ == 0:  # Fixed
                    s += 3  # Fx, Fz, M
                elif lager_typ == 1:  # Pinned
                    s += 2  # Fx, Fz
                elif lager_typ == 2:  # Roller
                    s += 1  # Fz
        
        # Count joint constraints
        c = 0
        for joint_data in self.joints.values():
            if len(joint_data.get('beams', [])) > 1:
                joint_type = joint_data.get('type', JOINT_FREE)
                if joint_type == JOINT_RIGID or joint_type == RIGID_CONNECTION:
                    c += 2  # Fx, Fz continuity (moment continuity is automatic)
                elif joint_type == JOINT_PIN:
                    c += 2  # Fx, Fz continuity (no moment continuity)
                # JOINT_FREE adds no constraints
        
        # For planar structures: 3n = s + c (where c includes joint constraints)
        required_constraints = 3 * n
        total_constraints = s + c
        
        if total_constraints < required_constraints:
            return False, f"Statically indeterminate: {total_constraints}<{required_constraints} (needs {required_constraints-total_constraints} more constraints)"
        elif total_constraints > required_constraints:
            return False, f"Statically overdetermined: {total_constraints}>{required_constraints} (has {total_constraints-required_constraints} extra constraints)"
        else:
            return True, f"Statically determinate: {total_constraints}={required_constraints}"
    
    def solve_connected_structure(self):
        """Solve the connected structure as a unified system"""
        # Clear previous solutions
        for beam in self.beams:
            beam.joint_forces_start = None
            beam.joint_forces_end = None
            beam.internal_forces_solved = False
        
        if len(self.beams) == 0:
            return False, "No beams in structure"
        
        # Check overall static determinacy
        is_determinate, status = self.check_overall_static_determinacy()
        
        if not is_determinate:
            return False, status
        
        # For a statically determinate connected structure, solve as unified system
        # This is a simplified approach - in practice would use matrix methods
        
        # Step 1: Identify supported beams (beams with supports)
        supported_beams = [beam for beam in self.beams if len(beam.lager) > 0]
        
        if len(supported_beams) == 0:
            return False, "No supports in structure - cannot solve"
        
        # Step 2: Solve supported beams first
        for beam in supported_beams:
            beam.solve_as_individual_beam()
            beam.internal_forces_solved = True
        
        # Step 3: Apply force continuity at all joints
        self._apply_joint_continuity()
        
        # Step 4: Mark all connected beams as solved
        for beam in self.beams:
            beam.internal_forces_solved = True
        
        return True, "Connected structure solved successfully"
    
    def _apply_joint_continuity(self):
        """Apply force and moment continuity at joints with proper load propagation"""
        for joint_key, joint_data in self.joints.items():
            beam_indices = joint_data.get('beams', [])
            joint_type = joint_data.get('type', JOINT_RIGID)
            
            if len(beam_indices) < 2:
                continue
                
            joint_pos = np.array(joint_key)
            
            # Calculate total forces at the joint from all connected beams
            total_N = 0
            total_Q = 0
            total_M = 0
            
            beam_forces = {}  # Store forces for each beam at this joint
            
            for beam_idx in beam_indices:
                beam = self.beams[beam_idx]
                
                # Determine if this is start or end of beam
                is_start = np.linalg.norm(beam.start - joint_pos) < self.joint_tolerance
                x_pos = 0.0 if is_start else beam.L
                
                # Get internal forces at joint
                N, Q, M = beam.schnittgroessen(x_pos)
                
                # Store beam forces
                beam_forces[beam_idx] = {
                    'N': N, 'Q': Q, 'M': M,
                    'is_start': is_start,
                    'has_loads': len(beam.punktlasten) > 0 or len(beam.streckenlasten) > 0,
                    'has_supports': len(beam.lager) > 0
                }
                
                # Sum forces (considering direction based on position)
                if is_start:
                    total_N -= N  # Forces at start point to the left
                    total_Q -= Q
                    total_M -= M
                else:
                    total_N += N  # Forces at end point to the right
                    total_Q += Q
                    total_M += M
            
            # Find the "driving" beam (beam with loads or supports)
            driving_beam_idx = None
            for beam_idx in beam_indices:
                beam_info = beam_forces[beam_idx]
                if beam_info['has_supports'] or beam_info['has_loads']:
                    driving_beam_idx = beam_idx
                    break
            
            if driving_beam_idx is None:
                driving_beam_idx = beam_indices[0]  # Default to first beam
            
            # Get reference forces from driving beam
            ref_forces = beam_forces[driving_beam_idx]
            
            # Apply continuity to other beams based on joint type
            for beam_idx in beam_indices:
                if beam_idx == driving_beam_idx:
                    continue
                    
                beam = self.beams[beam_idx]
                beam_info = beam_forces[beam_idx]
                
                if joint_type == JOINT_RIGID or joint_type == RIGID_CONNECTION:
                    # Full continuity: Forces must balance
                    if beam_info['is_start']:
                        beam.joint_forces_start = (-ref_forces['N'], -ref_forces['Q'], -ref_forces['M'])
                    else:
                        beam.joint_forces_end = (ref_forces['N'], ref_forces['Q'], ref_forces['M'])
                        
                elif joint_type == JOINT_PIN:
                    # Pin joint: Force continuity, no moment transfer
                    if beam_info['is_start']:
                        beam.joint_forces_start = (-ref_forces['N'], -ref_forces['Q'], 0.0)
                    else:
                        beam.joint_forces_end = (ref_forces['N'], ref_forces['Q'], 0.0)
                        
                elif joint_type == JOINT_FREE:
                    # Free connection: no force transfer
                    if beam_info['is_start']:
                        beam.joint_forces_start = (0.0, 0.0, 0.0)
                    else:
                        beam.joint_forces_end = (0.0, 0.0, 0.0)
    
    def _apply_force_continuity(self):
        """Apply force continuity conditions at joints"""
        for joint_key, joint_data in self.joints.items():
            beam_indices = joint_data.get('beams', [])
            joint_type = joint_data.get('type', JOINT_FREE)
            
            if len(beam_indices) < 2:
                continue
                
            joint_pos = np.array(joint_key)
            
            # For simplicity, calculate joint forces based on loads and apply them to unsupported beams
            # This is a simplified approach - proper analysis would solve the entire system simultaneously
            
            # Find beams with loads but no supports (these need force transfer)
            supported_beams = []
            unsupported_beams = []
            
            for beam_idx in beam_indices:
                beam = self.beams[beam_idx]
                if len(beam.lager) > 0:
                    supported_beams.append(beam)
                else:
                    unsupported_beams.append(beam)
            
            # If there's a supported beam, use it to provide forces to unsupported beams
            if len(supported_beams) > 0 and len(unsupported_beams) > 0:
                reference_beam = supported_beams[0]
                
                # Calculate the position on the reference beam
                ref_is_start = np.linalg.norm(reference_beam.start - joint_pos) < self.joint_tolerance
                if ref_is_start:
                    ref_x = 0.0
                else:
                    ref_x = reference_beam.L
                
                # Get internal forces at the joint from the reference beam
                N_ref, Q_ref, M_ref = reference_beam.schnittgroessen(ref_x)
                
                # Apply equilibrium forces to unsupported beams
                for unsupported_beam in unsupported_beams:
                    unsup_is_start = np.linalg.norm(unsupported_beam.start - joint_pos) < self.joint_tolerance
                    
                    if joint_type == JOINT_RIGID or joint_type == RIGID_CONNECTION:
                        # Full continuity: Forces must be equal and opposite for equilibrium
                        if unsup_is_start:
                            unsupported_beam.joint_forces_start = (-N_ref, -Q_ref, -M_ref)
                        else:
                            unsupported_beam.joint_forces_end = (N_ref, Q_ref, M_ref)
                    elif joint_type == JOINT_PIN:
                        # Only force continuity, no moment transfer
                        if unsup_is_start:
                            unsupported_beam.joint_forces_start = (-N_ref, -Q_ref, 0.0)
                        else:
                            unsupported_beam.joint_forces_end = (N_ref, Q_ref, 0.0)
                    # JOINT_FREE: no force transfer
            
            # Alternative: If no supported beams, try to distribute loads
            elif len(unsupported_beams) > 1:
                # For beams with loads but no supports, create artificial equilibrium
                total_loads = np.array([0.0, 0.0, 0.0])  # N, Q, M
                
                for beam in unsupported_beams:
                    # Sum up all loads on this beam
                    for pos_global, kraft_global in beam.punktlasten:
                        f_local = beam.global_to_local(kraft_global)
                        total_loads[0] += f_local[0]  # N
                        total_loads[1] += f_local[1]  # Q
                        # M contribution would need position calculation
                
                # Distribute the total loads among connected beams
                if len(unsupported_beams) > 1:
                    load_per_beam = total_loads / len(unsupported_beams)
                    
                    for beam in unsupported_beams:
                        beam_is_start = np.linalg.norm(beam.start - joint_pos) < self.joint_tolerance
                        
                        if joint_type == JOINT_RIGID or joint_type == RIGID_CONNECTION:
                            if beam_is_start:
                                beam.joint_forces_start = (-load_per_beam[0], -load_per_beam[1], -load_per_beam[2])
                            else:
                                beam.joint_forces_end = (load_per_beam[0], load_per_beam[1], load_per_beam[2])
                        elif joint_type == JOINT_PIN:
                            if beam_is_start:
                                beam.joint_forces_start = (-load_per_beam[0], -load_per_beam[1], 0.0)
                            else:
                                beam.joint_forces_end = (load_per_beam[0], load_per_beam[1], 0.0)
    
    def find_beam_at_point(self, point):
        """Find the beam closest to the given point"""
        if not self.beams:
            return None
            
        closest_beam = None
        min_distance = float('inf')
        
        for beam in self.beams:
            distance = beam.distance_to_point(point)
            if distance < min_distance:
                min_distance = distance
                closest_beam = beam
        
        # Only return beam if point is reasonably close (within 30 pixels)
        if min_distance < 30:
            return closest_beam
        return None
    
    def get_internal_force_continuity(self, joint_key):
        """Get internal force continuity at a joint based on connection type"""
        joint_data = self.joints.get(joint_key, {'type': JOINT_FREE})
        joint_type = joint_data['type']
        beam_indices = joint_data.get('beams', [])
        
        if len(beam_indices) < 2:
            return {}  # No continuity for single beam
            
        continuity = {}
        joint_pos = np.array(joint_key)
        
        for i, beam_idx in enumerate(beam_indices):
            beam = self.beams[beam_idx]
            
            # Determine if this is start or end of beam
            is_start = np.linalg.norm(beam.start - joint_pos) < self.joint_tolerance
            
            if joint_type == JOINT_RIGID or joint_type == RIGID_CONNECTION:
                # Full continuity: Q and M transfer
                continuity[f'beam_{beam_idx}'] = {
                    'position': 'start' if is_start else 'end',
                    'Q_continuous': True,
                    'M_continuous': True
                }
            elif joint_type == JOINT_PIN:
                # Pin connection: only Q transfers, M = 0
                continuity[f'beam_{beam_idx}'] = {
                    'position': 'start' if is_start else 'end',
                    'Q_continuous': True,
                    'M_continuous': False,
                    'M_value': 0  # Moment release
                }
            else:  # JOINT_FREE
                # No connection: independent beams
                continuity[f'beam_{beam_idx}'] = {
                    'position': 'start' if is_start else 'end',
                    'Q_continuous': False,
                    'M_continuous': False
                }
                
        return continuity
    
    def get_merged_polygons(self):
        """Generate merged polygons for connected beams"""
        # This is a simplified approach - in practice you'd use proper polygon union algorithms
        all_corners = []
        
        for beam in self.beams:
            corners = beam.get_corners()
            all_corners.extend(corners)
            
        return all_corners
    
    def draw_continuous_internal_forces(self, surf, scale_factor=0.01):
        """Draw continuous internal force diagrams across the entire connected structure"""
        if len(self.beams) == 0:
            return
            
        # Check if structure is solved
        solved, status = self.solve_connected_structure()
        if not solved:
            return
            
        # Generate continuous points across all beams in order
        all_pts_N, all_pts_Q, all_pts_M = [], [], []
        all_beam_line_points = []
        
        for beam_idx, beam in enumerate(self.beams):
            if beam.L == 0:
                continue
                
            # Get segments for this beam
            segments = beam.get_segments()
            
            # Generate points for this beam
            for i in range(len(segments) - 1):
                x_start = segments[i]
                x_end = segments[i + 1]
                
                num_points = max(5, int((x_end - x_start) / beam.L * 50))
                
                for j in range(num_points + 1):
                    if j == num_points and i < len(segments) - 2:
                        continue  # Skip last point except for last segment
                        
                    t = j / num_points if num_points > 0 else 0
                    x_local = x_start + t * (x_end - x_start)
                    
                    # Convert to world coordinates
                    world_pos = beam.world_point(x_local)
                    
                    # Get internal forces
                    N, Q, M = beam.schnittgroessen(x_local)
                    
                    # Store points
                    all_beam_line_points.append(world_pos)
                    all_pts_N.append(world_pos + beam.e_z * N * scale_factor)
                    all_pts_Q.append(world_pos + beam.e_z * Q * scale_factor)
                    all_pts_M.append(world_pos + beam.e_z * M * scale_factor * 0.01)
        
        # Check if diagrams have non-zero values
        has_N_values = any(abs(N_pt[1] - beam_pt[1]) > 0.1 for N_pt, beam_pt in zip(all_pts_N, all_beam_line_points))
        has_Q_values = any(abs(Q_pt[1] - beam_pt[1]) > 0.1 for Q_pt, beam_pt in zip(all_pts_Q, all_beam_line_points))
        has_M_values = any(abs(M_pt[1] - beam_pt[1]) > 0.1 for M_pt, beam_pt in zip(all_pts_M, all_beam_line_points))
        
        # Draw continuous filled areas
        if len(all_pts_N) > 1 and has_N_values:
            n_polygon = all_pts_N + list(reversed(all_beam_line_points))
            draw_transparent_polygon(surf, COLORS['N'], n_polygon, 70)
            
        if len(all_pts_Q) > 1 and has_Q_values:
            q_polygon = all_pts_Q + list(reversed(all_beam_line_points))
            draw_transparent_polygon(surf, COLORS['Q'], q_polygon, 70)
            
        if len(all_pts_M) > 1 and has_M_values:
            m_polygon = all_pts_M + list(reversed(all_beam_line_points))
            draw_transparent_polygon(surf, COLORS['M'], m_polygon, 70)
        
        # Draw continuous lines
        if len(all_pts_N) > 1 and has_N_values:
            pygame.draw.lines(surf, COLORS['N'], False, all_pts_N, 2)
        if len(all_pts_Q) > 1 and has_Q_values:
            pygame.draw.lines(surf, COLORS['Q'], False, all_pts_Q, 2)
        if len(all_pts_M) > 1 and has_M_values:
            pygame.draw.lines(surf, COLORS['M'], False, all_pts_M, 2)
            
        # Add labels at key points
        font_legend = get_font('legend')
        
        if len(all_pts_N) > 10 and has_N_values:
            mid_idx = len(all_pts_N) // 2
            n_pos = all_pts_N[mid_idx]
            beam_pos = all_beam_line_points[mid_idx]
            
            if n_pos[1] < beam_pos[1]:
                text_pos = n_pos + np.array([0, -20])
            else:
                text_pos = n_pos + np.array([0, 10])
                
            n_text = font_legend.render("N(x) - Continuous", True, COLORS['N'])
            surf.blit(n_text, text_pos.astype(int))
        
        if len(all_pts_Q) > 10 and has_Q_values:
            mid_idx = len(all_pts_Q) // 2
            q_pos = all_pts_Q[mid_idx]
            beam_pos = all_beam_line_points[mid_idx]
            
            if q_pos[1] < beam_pos[1]:
                text_pos = q_pos + np.array([0, -20])
            else:
                text_pos = q_pos + np.array([0, 10])
                
            q_text = font_legend.render("Q(x) - Continuous", True, COLORS['Q'])
            surf.blit(q_text, text_pos.astype(int))
        
        if len(all_pts_M) > 10 and has_M_values:
            mid_idx = len(all_pts_M) // 2
            m_pos = all_pts_M[mid_idx]
            beam_pos = all_beam_line_points[mid_idx]
            
            if m_pos[1] < beam_pos[1]:
                text_pos = m_pos + np.array([0, -20])
            else:
                text_pos = m_pos + np.array([0, 10])
                
            m_text = font_legend.render("M(x) - Continuous", True, COLORS['M'])
            surf.blit(m_text, text_pos.astype(int))
    
    def draw(self, surf):
        """Draw the structure with proper joint connections"""
        # First, draw merged polygons for rigid connections
        self.draw_merged_rigid_connections(surf)
        
        # Draw individual beams (for non-rigid connections) - basic version without supports
        for beam in self.beams:
            beam.draw_basic(surf)
            
        # Draw joint connections with type-specific colors
        for joint_pos, joint_data in self.joints.items():
            beam_indices = joint_data.get('beams', [])
            joint_type = joint_data.get('type', JOINT_FREE)
            
            if len(beam_indices) > 1:  # Only draw joints with multiple beams
                joint_point = np.array(joint_pos, dtype=float)
                
                if joint_type == RIGID_CONNECTION:
                    # Rigid connection: no visible joint symbol, just continuous polygon
                    # This will be handled by drawing merged beam polygons
                    continue
                elif joint_type == JOINT_RIGID:
                    # Rigid joint: filled circle with outline (like support icons)
                    pygame.draw.circle(surf, COLORS['symbol_bg'], joint_point.astype(int), 15)  # Background
                    pygame.draw.circle(surf, COLORS['joint_rigid'], joint_point.astype(int), 12, 2)  # Outline
                    type_text = "R"
                elif joint_type == JOINT_PIN:
                    # Pin joint: circle outline on filled circle (like support icons)
                    pygame.draw.circle(surf, COLORS['symbol_bg'], joint_point.astype(int), 15)  # Background
                    pygame.draw.circle(surf, COLORS['joint_pin'], joint_point.astype(int), 12, 2)    # Outline
                    pygame.draw.circle(surf, COLORS['joint_pin'], joint_point.astype(int), 6, 2)     # Inner circle
                    type_text = "P"
                else:  # JOINT_FREE
                    # Free joint: just outline circle
                    pygame.draw.circle(surf, COLORS['symbol_bg'], joint_point.astype(int), 15)  # Background
                    pygame.draw.circle(surf, COLORS['joint_free'], joint_point.astype(int), 12, 2)   # Outline only
                    type_text = "F"
                
                # Draw joint type indicator (except for rigid connection)
                font = get_font(12)
                text = font.render(type_text, True, COLORS['ui_text'])
                text_rect = text.get_rect(center=joint_point.astype(int))
                surf.blit(text, text_rect)
    
    def draw_merged_rigid_connections(self, surf):
        """Draw merged polygons for beams connected with rigid connections"""
        # For now, this is a simplified implementation that just draws individual beams
        # A more sophisticated implementation would merge the beam polygons properly
        # This could be enhanced later with proper polygon merging algorithms
        pass
    
    def find_beam_at_point(self, point):
        """Find the beam closest to a point"""
        closest_beam = None
        min_distance = float('inf')
        
        for beam in self.beams:
            distance = beam.distance_to_point(point)
            if distance < min_distance and distance < 30:
                min_distance = distance
                closest_beam = beam
                
        return closest_beam

class Beam:
    """Individual beam element in a connected structure with full structural analysis"""
    def __init__(self, start, end, beam_id=0):
        self.beam_id = beam_id
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.L = np.linalg.norm(self.end - self.start)
        
        if self.L > 0:
            self.e_x = (self.end - self.start) / self.L  # Unit vector along beam
            self.e_z = np.array([-self.e_x[1], self.e_x[0]])  # Perpendicular unit vector
        else:
            self.e_x = np.array([1, 0])
            self.e_z = np.array([0, 1])
            
        # Loads (same as original)
        self.punktlasten = []  # Point loads: [(position, force_vector), ...]
        self.streckenlasten = []  # Line loads: [(start, end, end_amp, start_amp), ...]
        self.lager = {}  # Supports: {position: support_type}
        
        # For connected beam analysis
        self.connected_structure = None
        self.joint_forces_start = None  # (N, Q, M) at start - forces from connected beams
        self.joint_forces_end = None    # (N, Q, M) at end - forces from connected beams
        self.internal_forces_solved = False
        
        # Reaction forces (A,B,C,D for fixed/pinned, E,F for roller)
        self.A, self.B, self.C, self.D = 0, 0, 0, 0
        self.E, self.F = 0, 0
        
    def global_to_local(self, v):
        """Convert global vector to local coordinates"""
        return np.array([np.dot(v, self.e_x), np.dot(v, self.e_z)])

    def world_point(self, x):
        """Convert local x-coordinate to world coordinates"""
        return self.start + x * self.e_x

    def snap_to_beam(self, pos):
        """Snap a point to the nearest point on the beam"""
        vec_to_point = pos - self.start
        projection_length = np.dot(vec_to_point, self.e_x)
        projection_length = max(0, min(self.L, projection_length))
        return self.start + projection_length * self.e_x

    def add_punktlast(self, pos, richtung):
        """Add point load"""
        snapped_pos = self.snap_to_beam(pos)
        self.punktlasten.append((snapped_pos, richtung))

    def add_streckenlast(self, start_pos, end_pos, end_amplitude, start_amplitude=None):
        """Add line load"""
        snapped_start = self.snap_to_beam(start_pos)
        snapped_end = self.snap_to_beam(end_pos)
        
        end_amplitude_z = np.dot(end_amplitude, self.e_z)
        
        if start_amplitude is not None:
            start_amplitude_z = np.dot(start_amplitude, self.e_z)
        else:
            start_amplitude_z = end_amplitude_z
        
        self.streckenlasten.append((snapped_start, snapped_end, end_amplitude_z, start_amplitude_z))

    def add_lager(self, pos):
        """Add/toggle support"""
        dist_start = np.linalg.norm(pos - self.start)
        dist_end = np.linalg.norm(pos - self.end)
        
        if dist_start < dist_end:
            lager_pos = "start"
            snap_pos = self.start
        else:
            lager_pos = "end"
            snap_pos = self.end
            
        # Cycle through support types: 0->1->2->None->0...
        if lager_pos in self.lager:
            current_type = self.lager[lager_pos]
            if current_type == 2:  # Roller -> remove
                del self.lager[lager_pos]
            else:
                self.lager[lager_pos] = current_type + 1
        else:
            self.lager[lager_pos] = 0  # Fixed support
            
        return snap_pos

    def pruefe_statische_bestimmtheit(self):
        """Check static determinacy"""
        n = 1  # One beam
        v = 0  # No hinges (for single beam)
        s = 0  # Support reactions
        
        for lager_pos, lager_typ in self.lager.items():
            if lager_typ == 0:  # Fixed
                s += 3  # Fx, Fz, M
            elif lager_typ == 1:  # Pinned
                s += 2  # Fx, Fz
            elif lager_typ == 2:  # Roller
                s += 1  # Fz
        
        required_reactions = 3 * n  # = 3 for one beam
        
        if s < required_reactions:
            return False, f"Statically indeterminate: {s}<{required_reactions}"
        elif s > required_reactions:
            return False, f"Statically overdetermined: {s}>{required_reactions}"
        else:
            return True, f"Statically determinate: {s}={required_reactions}"

    def get_segments(self):
        """Create segment division based on load positions"""
        segments = [0, self.L]
        
        # Add point loads
        for pos_global, _ in self.punktlasten:
            x_l = np.dot(pos_global - self.start, self.e_x)
            segments.append(x_l)
        
        # Add line loads
        for start_pos, end_pos, end_amplitude, start_amplitude in self.streckenlasten:
            x1 = np.dot(start_pos - self.start, self.e_x)
            x2 = np.dot(end_pos - self.start, self.e_x)
            segments.extend([x1, x2])
        
        segments = sorted(list(set(segments)))
        return segments

    def berechne_lagerreaktionen(self):
        """Calculate support reactions"""
        ist_bestimmt, _ = self.pruefe_statische_bestimmtheit()
        if not ist_bestimmt:
            return {}
            
        lager_reaktionen = {}
        
        # Collect all external loads
        sum_Fx = sum_Fz = sum_M_start = 0
        
        # Point loads
        for pos_global, kraft_global in self.punktlasten:
            x_l = np.dot(pos_global - self.start, self.e_x)
            f_local = self.global_to_local(kraft_global)
            fx_punkt = f_local[0]
            fz_punkt = f_local[1]
            
            sum_Fx += fx_punkt
            sum_Fz += fz_punkt
            sum_M_start += fz_punkt * x_l  # Moment about start
        
        # Line loads
        for start_pos, end_pos, end_amplitude, start_amplitude in self.streckenlasten:
            x1 = np.dot(start_pos - self.start, self.e_x)
            x2 = np.dot(end_pos - self.start, self.e_x)
            
            length = abs(x2 - x1)
            if length > 0:
                # Resultant force and its position
                F_res = (start_amplitude + end_amplitude) * length / 2
                x_res = x1 + length/2 + (end_amplitude - start_amplitude) * length / (6 * (start_amplitude + end_amplitude)) if (start_amplitude + end_amplitude) != 0 else x1 + length/2
                
                sum_Fz += F_res
                sum_M_start += F_res * x_res
        
        # Solve equilibrium equations based on support configuration
        if len(self.lager) == 0:
            return {}
            
        elif len(self.lager) == 1:
            # Single support - check type
            lager_pos = list(self.lager.keys())[0]
            lager_typ = self.lager[lager_pos]
            
            if lager_typ == 0:  # Fixed support
                fx = -sum_Fx
                fz = -sum_Fz
                m = -sum_M_start if lager_pos == "start" else -(sum_M_start - sum_Fz * self.L)
                lager_reaktionen[lager_pos] = (fx, fz, m)
            else:
                # Pinned or roller - cannot carry all reactions for single support
                lager_reaktionen[lager_pos] = (0, -sum_Fz if lager_typ >= 1 else 0, 0)
                
        elif len(self.lager) == 2:
            # Two supports
            lager_positionen = list(self.lager.keys())
            
            if "start" in lager_positionen and "end" in lager_positionen:
                # Simple beam with supports at both ends
                # Solve using equilibrium equations
                # Sum of moments about start = 0
                fz_end = sum_M_start / self.L
                fz_start = -sum_Fz - fz_end
                
                # Horizontal forces
                fx_start = -sum_Fx
                fx_end = 0
                
                # Moments (only for fixed supports)
                m_start = 0 if self.lager["start"] != 0 else 0  # Calculate if needed
                m_end = 0 if self.lager["end"] != 0 else 0
                
                lager_reaktionen["start"] = (fx_start, fz_start, m_start)
                lager_reaktionen["end"] = (fx_end, fz_end, m_end)
        
        return lager_reaktionen

    def schnittgroessen(self, x):
        """Calculate internal forces at position x, considering joint forces and boundary conditions"""
        # Only solve if part of a solved connected structure
        if self.connected_structure:
            if not self.internal_forces_solved:
                return 0, 0, 0  # Structure not solved yet
        else:
            # Individual beam analysis
            ist_bestimmt, _ = self.pruefe_statische_bestimmtheit()
            if not ist_bestimmt:
                return 0, 0, 0
        
        N = Q = M = 0
        
        # 1. JOINT FORCES: Apply forces from connected beams
        if self.joint_forces_start is not None:
            N_joint, Q_joint, M_joint = self.joint_forces_start
            # Joint forces act throughout the beam
            N += N_joint
            Q += Q_joint
            M += M_joint
        
        if self.joint_forces_end is not None:
            N_joint, Q_joint, M_joint = self.joint_forces_end
            # Forces at end only affect sections before the end
            if x < self.L:
                N += N_joint
                Q += Q_joint
                M -= Q_joint * (self.L - x)  # Moment arm
                M += M_joint
        
        # 2. SUPPORT REACTIONS (if beam has supports)
        if len(self.lager) > 0:
            lager_reaktionen = self.berechne_lagerreaktionen()
            
            for lager_pos, (fx, fz, m_lager) in lager_reaktionen.items():
                if lager_pos == "start":
                    x_l = 0
                elif lager_pos == "end":
                    x_l = self.L
                else:
                    continue
                    
                if x_l > x:
                    N += fx
                    Q += fz
                    M -= fz * (x_l - x)
                    
                    if lager_pos in self.lager and self.lager[lager_pos] == 0:
                        M += m_lager
        
        # 3. POINT LOADS (to the right of cut)
        for pos_global, kraft_global in self.punktlasten:
            x_l = np.dot(pos_global - self.start, self.e_x)
            
            if x_l > x:
                f_local = self.global_to_local(kraft_global)
                fx_punkt = f_local[0]
                fz_punkt = f_local[1]
                
                N += fx_punkt
                Q += fz_punkt
                M -= fz_punkt * (x_l - x)
        
        # 4. LINE LOADS (to the right of cut)
        for start_pos, end_pos, end_amplitude, start_amplitude in self.streckenlasten:
            x1 = np.dot(start_pos - self.start, self.e_x)
            x2 = np.dot(end_pos - self.start, self.e_x)
            x1, x2 = min(x1, x2), max(x1, x2)
            
            if x2 <= x:
                continue
                
            x_left = max(x, x1)
            x_right = x2
            
            if x_left < x_right:
                length_total = x2 - x1
                if length_total > 0:
                    q1 = start_amplitude / length_total
                    q2 = end_amplitude / length_total
                    
                    l_active = x_right - x_left
                    xi1 = x_left - x1
                    xi2 = x_right - x1
                    
                    F_res = q1 * l_active + (q2-q1) * (xi2**2 - xi1**2) / (2 * length_total)
                    
                    if abs(F_res) > 1e-12:
                        moment_integral = q1 * (xi2**2 - xi1**2) / 2 + (q2-q1) * (xi2**3 - xi1**3) / (3 * length_total)
                        x_centroid = moment_integral / F_res + x1
                    else:
                        x_centroid = (x_left + x_right) / 2
                    
                    Q += F_res
                    M -= F_res * (x_centroid - x)
        
        # 5. BOUNDARY CONDITIONS
        # Free ends: M = 0
        if abs(x - 0) < 1e-6 and "start" not in self.lager:
            # Check if start is connected to rigid joint
            if self.connected_structure:
                start_has_rigid_joint = any(
                    np.linalg.norm(np.array(joint_key) - self.start) < self.connected_structure.joint_tolerance
                    and joint_data.get('type') in [JOINT_RIGID, RIGID_CONNECTION]
                    for joint_key, joint_data in self.connected_structure.joints.items()
                )
                if not start_has_rigid_joint:
                    M = 0  # Free end
            else:
                M = 0  # Free end
                
        if abs(x - self.L) < 1e-6 and "end" not in self.lager:
            # Check if end is connected to rigid joint
            if self.connected_structure:
                end_has_rigid_joint = any(
                    np.linalg.norm(np.array(joint_key) - self.end) < self.connected_structure.joint_tolerance
                    and joint_data.get('type') in [JOINT_RIGID, RIGID_CONNECTION]
                    for joint_key, joint_data in self.connected_structure.joints.items()
                )
                if not end_has_rigid_joint:
                    M = 0  # Free end
            else:
                M = 0  # Free end
        
        # Pin and roller supports: M = 0
        for lager_pos, lager_typ in self.lager.items():
            x_lager = 0 if lager_pos == "start" else self.L
            if abs(x - x_lager) < 1e-6 and lager_typ in [1, 2]:  # Pin or roller
                M = 0
        
        return N, Q, M

    def solve_as_individual_beam(self):
        """Solve this beam as an individual element with joint forces"""
        # This method prepares the beam for structural analysis
        # In a connected structure, joint forces will be applied after solving
        self.internal_forces_solved = True
        return True

    def get_corners(self):
        """Get the four corner points of the beam rectangle"""
        thickness = 20
        half_thickness = thickness / 2
        offset = self.e_z * half_thickness
        
        return [
            self.start + offset,
            self.start - offset,
            self.end - offset,
            self.end + offset
        ]
    
    def distance_to_point(self, point):
        """Calculate distance from point to beam centerline"""
        if self.L == 0:
            return np.linalg.norm(np.array(point) - self.start)
            
        v = np.array(point) - self.start
        proj_length = np.dot(v, self.e_x)
        proj_length = max(0, min(self.L, proj_length))
        closest_on_beam = self.start + proj_length * self.e_x
        return np.linalg.norm(np.array(point) - closest_on_beam)
    
    def draw_lager(self, surf, pos, lager_typ):
        """Draw support symbols at given position"""
        offset = np.array([-2, -2])
        centered_pos = pos + offset
        
        if lager_typ == 0:  # Fixed support
            pygame.draw.circle(surf, COLORS['symbol_bg'], pos.astype(int), 20)
            
            start_pos = centered_pos + np.array([0, -10])
            end_pos = centered_pos + np.array([0, 10])
            pygame.draw.line(surf, COLORS['symbol_line'], start_pos.astype(int), end_pos.astype(int), 2)
            
            base_y = centered_pos[1] + 10
            for i in range(5):
                x_pos = centered_pos[0] - 6 + i * 3
                pygame.draw.line(surf, COLORS['symbol_line'], (x_pos, base_y), (x_pos + 2, base_y + 5), 2)
                
        elif lager_typ == 1:  # Pinned support
            pygame.draw.circle(surf, COLORS['symbol_bg'], pos.astype(int), 20)
            
            symbol_offset = np.array([1, 0])
            symbol_pos = centered_pos + symbol_offset
            
            triangle_points = [
                (symbol_pos[0], symbol_pos[1] - 8),
                (symbol_pos[0] - 9, symbol_pos[1] + 6),
                (symbol_pos[0] + 9, symbol_pos[1] + 6)
            ]
            pygame.draw.polygon(surf, COLORS['symbol_line'], triangle_points, 2)
            
            circle_offset = 1
            pygame.draw.circle(surf, COLORS['symbol_line'], (int(symbol_pos[0] + circle_offset), int(symbol_pos[1] - 8)), 3, 2)
            
            base_y = symbol_pos[1] + 6
            for i in range(5):
                x_pos = symbol_pos[0] - 6 + i * 3
                pygame.draw.line(surf, COLORS['symbol_line'], (x_pos, base_y), (x_pos + 2, base_y + 5), 2)
                
        elif lager_typ == 2:  # Roller support
            pygame.draw.circle(surf, COLORS['symbol_bg'], pos.astype(int), 20)
            
            symbol_offset = np.array([1, 0])
            symbol_pos = centered_pos + symbol_offset
            
            triangle_points = [
                (symbol_pos[0], symbol_pos[1] - 8),
                (symbol_pos[0] - 9, symbol_pos[1] + 6),
                (symbol_pos[0] + 9, symbol_pos[1] + 6)
            ]
            pygame.draw.polygon(surf, COLORS['symbol_line'], triangle_points, 2)
            
            circle_offset = 1
            pygame.draw.circle(surf, COLORS['symbol_line'], (int(symbol_pos[0] + circle_offset), int(symbol_pos[1] - 8)), 3, 2)
            
            base_y = symbol_pos[1] + 11
            for i in range(5):
                x_pos = symbol_pos[0] - 6 + i * 3
                pygame.draw.line(surf, COLORS['symbol_line'], (x_pos, base_y), (x_pos + 2, base_y + 5), 2)

    def draw_basic(self, surf):
        """Draw just the beam rectangle without loads or supports (for structure drawing)"""
        corners = self.get_corners()
        pygame.draw.polygon(surf, COLORS['beam'], corners)
        
        # Draw beam ID at center
        center = (self.start + self.end) / 2
        font = get_font('values')
        text = font.render(f"B{self.beam_id}", True, COLORS['ui_text'])
        text_rect = text.get_rect(center=center.astype(int))
        surf.blit(text, text_rect)

    def draw_loads_and_reactions(self, surf):
        """Draw loads, reactions, and supports without the basic beam shape"""
        # Draw point loads
        for pos_global, kraft_global in self.punktlasten:
            tip = pos_global + kraft_global
            pygame.draw.line(surf, COLORS['force_line'], pos_global, tip, 2)
            
            # Arrow head
            arrow_length = 8
            arrow_angle = 0.3
            kraft_norm = np.linalg.norm(kraft_global)
            if kraft_norm > 0:
                kraft_unit = kraft_global / kraft_norm
                left_wing = tip - arrow_length * (kraft_unit * np.cos(arrow_angle) + 
                           np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                right_wing = tip - arrow_length * (kraft_unit * np.cos(arrow_angle) - 
                            np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                arrow_points = [tip, left_wing, right_wing]
                pygame.draw.polygon(surf, COLORS['force_line'], arrow_points)
            
            # Force value
            font_values = get_font('values')
            text = font_values.render(f"{kraft_norm:.0f}N", True, COLORS['force_text'])
            
            if kraft_norm > 0:
                kraft_unit = kraft_global / kraft_norm
                text_offset = kraft_unit * 15
                text_pos = tip + text_offset
            else:
                text_pos = tip + np.array([5, -15])
                
            text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            surf.blit(text, text_rect)
            
        # Draw line loads
        for start_pos, end_pos, end_amplitude, start_amplitude in self.streckenlasten:
            kraft_vektor_end = self.e_z * end_amplitude
            kraft_vektor_start = self.e_z * start_amplitude
            
            # Draw trapezoid
            if abs(start_amplitude) > 1e-6 or abs(end_amplitude) > 1e-6:
                rect_points = [start_pos, end_pos, end_pos + kraft_vektor_end, start_pos + kraft_vektor_start]
                pygame.draw.polygon(surf, COLORS['force_line'], rect_points, 2)
                draw_transparent_polygon(surf, COLORS['force_line'], rect_points, 50)
            
            # Draw arrows
            laenge = np.linalg.norm(end_pos - start_pos)
            num_arrows = max(3, int(laenge / 30))
            
            for i in range(num_arrows):
                t = i / (num_arrows - 1) if num_arrows > 1 else 0
                arrow_start = start_pos + t * (end_pos - start_pos)
                
                current_amplitude = start_amplitude + t * (end_amplitude - start_amplitude)
                kraft_vektor = self.e_z * current_amplitude
                arrow_end = arrow_start + kraft_vektor
                
                if np.linalg.norm(kraft_vektor) > 1e-6:
                    pygame.draw.line(surf, COLORS['force_line'], arrow_start, arrow_end, 2)
                    
                    arrow_length = 8
                    arrow_angle = 0.3
                    kraft_norm = np.linalg.norm(kraft_vektor)
                    if kraft_norm > 0:
                        kraft_unit = kraft_vektor / kraft_norm
                        left_wing = arrow_end - arrow_length * (kraft_unit * np.cos(arrow_angle) + 
                                   np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                        right_wing = arrow_end - arrow_length * (kraft_unit * np.cos(arrow_angle) - 
                                    np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                        arrow_points = [arrow_end, left_wing, right_wing]
                        pygame.draw.polygon(surf, COLORS['force_line'], arrow_points)
            
            # Force values
            font_values = get_font('values')
            if abs(start_amplitude - end_amplitude) < 1e-6:
                # Uniform load
                kraft_pro_meter = abs(end_amplitude) / (laenge / 1000) if laenge > 0 else 0
                text = font_values.render(f"{kraft_pro_meter:.0f}N/m", True, COLORS['force_text'])
                
                mid_pos = (start_pos + end_pos) / 2
                if abs(end_amplitude) > 0:
                    kraft_unit = kraft_vektor_end / np.linalg.norm(kraft_vektor_end)
                    text_offset = kraft_unit * 15
                    text_pos = mid_pos + kraft_vektor_end + text_offset
                else:
                    text_pos = mid_pos + np.array([5, -5])
                text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text, text_rect)
            else:
                # Variable load
                start_kraft_pro_meter = abs(start_amplitude) / (laenge / 1000) if laenge > 0 else 0
                end_kraft_pro_meter = abs(end_amplitude) / (laenge / 1000) if laenge > 0 else 0
                
                if abs(start_amplitude) > 1e-6:
                    text_start = font_values.render(f"{start_kraft_pro_meter:.0f}N/m", True, COLORS['force_text'])
                    if abs(start_amplitude) > 0:
                        kraft_unit = kraft_vektor_start / np.linalg.norm(kraft_vektor_start)
                        text_offset = kraft_unit * 15
                        text_pos = start_pos + kraft_vektor_start + text_offset
                    else:
                        text_pos = start_pos + np.array([5, -5])
                    text_rect = text_start.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                    surf.blit(text_start, text_rect)
                
                if abs(end_amplitude) > 1e-6:
                    text_end = font_values.render(f"{end_kraft_pro_meter:.0f}N/m", True, COLORS['force_text'])
                    if abs(end_amplitude) > 0:
                        kraft_unit = kraft_vektor_end / np.linalg.norm(kraft_vektor_end)
                        text_offset = kraft_unit * 15
                        text_pos = end_pos + kraft_vektor_end + text_offset
                    else:
                        text_pos = end_pos + np.array([5, -5])
                    text_rect = text_end.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                    surf.blit(text_end, text_rect)
            
        # Draw support reactions
        ist_bestimmt, _ = self.pruefe_statische_bestimmtheit()
        if ist_bestimmt and len(self.lager) > 0:
            lager_reaktionen = self.berechne_lagerreaktionen()
            for lager_pos, (fx, fz, m_lager) in lager_reaktionen.items():
                if lager_pos == "start":
                    pos = self.start
                elif lager_pos == "end":
                    pos = self.end
                else:
                    continue
                    
                # Vertical reaction
                if abs(fz) > 0.1:
                    kraft_vektor = self.e_z * fz * 0.5
                    tip = pos + kraft_vektor
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, 2)
                    
                    arrow_length = 6
                    if abs(fz) > 0:
                        fz_unit = kraft_vektor / np.linalg.norm(kraft_vektor)
                        left_wing = tip - arrow_length * (fz_unit * np.cos(0.3) + 
                                   np.array([-fz_unit[1], fz_unit[0]]) * np.sin(0.3))
                        right_wing = tip - arrow_length * (fz_unit * np.cos(0.3) - 
                                    np.array([-fz_unit[1], fz_unit[0]]) * np.sin(0.3))
                        arrow_points = [tip, left_wing, right_wing]
                        pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)
                    
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fz:.0f}N", True, COLORS['reaction_text'])
                    text_pos = tip + np.array([5, -10])
                    surf.blit(text, text_pos.astype(int))
                    
                # Horizontal reaction
                if abs(fx) > 0.1:
                    kraft_vektor = self.e_x * fx * 0.5
                    tip = pos + kraft_vektor
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, 2)
                    
                    if abs(fx) > 0:
                        fx_unit = kraft_vektor / np.linalg.norm(kraft_vektor)
                        left_wing = tip - 6 * (fx_unit * np.cos(0.3) + 
                                   np.array([-fx_unit[1], fx_unit[0]]) * np.sin(0.3))
                        right_wing = tip - 6 * (fx_unit * np.cos(0.3) - 
                                    np.array([-fx_unit[1], fx_unit[0]]) * np.sin(0.3))
                        arrow_points = [tip, left_wing, right_wing]
                        pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)
                    
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fx:.0f}N", True, COLORS['force_text'])
                    text_pos = tip + np.array([5, 5])
                    surf.blit(text, text_pos.astype(int))
                    
        # Draw supports last to overlay everything (but not over joints)
        for lager_pos, lager_typ in self.lager.items():
            if lager_pos == "start":
                self.draw_lager(surf, self.start, lager_typ)
            elif lager_pos == "end":
                self.draw_lager(surf, self.end, lager_typ)

    def draw(self, surf):
        """Draw beam with all loads, forces, and supports"""
        # Draw beam rectangle
        corners = self.get_corners()
        pygame.draw.polygon(surf, COLORS['beam'], corners)
        
        # Draw local coordinate system at start
        x_axis_end = self.start + self.e_x * 40
        z_axis_end = self.start + self.e_z * 40
        
        pygame.draw.line(surf, COLORS['x_axis'], self.start, x_axis_end, 2)
        pygame.draw.line(surf, COLORS['z_axis'], self.start, z_axis_end, 2)
        
        font_axis = get_font('axis')
        x_text = font_axis.render("x", True, COLORS['x_axis'])
        z_text = font_axis.render("z", True, COLORS['z_axis'])
        surf.blit(x_text, (x_axis_end + np.array([5, -10])).astype(int))
        surf.blit(z_text, (z_axis_end + np.array([5, -10])).astype(int))
        
        # Draw beam ID
        center = (self.start + self.end) / 2
        font = get_font('values')
        text = font.render(f"B{self.beam_id}", True, COLORS['ui_text'])
        text_rect = text.get_rect(center=center.astype(int))
        surf.blit(text, text_rect)
        
        # Draw point loads
        for pos_global, kraft_global in self.punktlasten:
            tip = pos_global + kraft_global
            pygame.draw.line(surf, COLORS['force_line'], pos_global, tip, 2)
            
            # Arrow head
            arrow_length = 8
            arrow_angle = 0.3
            kraft_norm = np.linalg.norm(kraft_global)
            if kraft_norm > 0:
                kraft_unit = kraft_global / kraft_norm
                left_wing = tip - arrow_length * (kraft_unit * np.cos(arrow_angle) + 
                           np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                right_wing = tip - arrow_length * (kraft_unit * np.cos(arrow_angle) - 
                            np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                arrow_points = [tip, left_wing, right_wing]
                pygame.draw.polygon(surf, COLORS['force_line'], arrow_points)
            
            # Force value
            font_values = get_font('values')
            text = font_values.render(f"{kraft_norm:.0f}N", True, COLORS['force_text'])
            
            if kraft_norm > 0:
                kraft_unit = kraft_global / kraft_norm
                text_offset = kraft_unit * 15
                text_pos = tip + text_offset
            else:
                text_pos = tip + np.array([5, -15])
                
            text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            surf.blit(text, text_rect)
            
        # Draw line loads
        for start_pos, end_pos, end_amplitude, start_amplitude in self.streckenlasten:
            kraft_vektor_end = self.e_z * end_amplitude
            kraft_vektor_start = self.e_z * start_amplitude
            
            # Draw trapezoid
            if abs(start_amplitude) > 1e-6 or abs(end_amplitude) > 1e-6:
                rect_points = [start_pos, end_pos, end_pos + kraft_vektor_end, start_pos + kraft_vektor_start]
                pygame.draw.polygon(surf, COLORS['force_line'], rect_points, 2)
                draw_transparent_polygon(surf, COLORS['force_line'], rect_points, 50)
            
            # Draw arrows
            laenge = np.linalg.norm(end_pos - start_pos)
            num_arrows = max(3, int(laenge / 30))
            
            for i in range(num_arrows):
                t = i / (num_arrows - 1) if num_arrows > 1 else 0
                arrow_start = start_pos + t * (end_pos - start_pos)
                
                current_amplitude = start_amplitude + t * (end_amplitude - start_amplitude)
                kraft_vektor = self.e_z * current_amplitude
                arrow_end = arrow_start + kraft_vektor
                
                if np.linalg.norm(kraft_vektor) > 1e-6:
                    pygame.draw.line(surf, COLORS['force_line'], arrow_start, arrow_end, 2)
                    
                    arrow_length = 8
                    arrow_angle = 0.3
                    kraft_norm = np.linalg.norm(kraft_vektor)
                    if kraft_norm > 0:
                        kraft_unit = kraft_vektor / kraft_norm
                        left_wing = arrow_end - arrow_length * (kraft_unit * np.cos(arrow_angle) + 
                                   np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                        right_wing = arrow_end - arrow_length * (kraft_unit * np.cos(arrow_angle) - 
                                    np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                        arrow_points = [arrow_end, left_wing, right_wing]
                        pygame.draw.polygon(surf, COLORS['force_line'], arrow_points)
            
            # Force values
            font_values = get_font('values')
            if abs(start_amplitude - end_amplitude) < 1e-6:
                # Uniform load
                kraft_pro_meter = abs(end_amplitude) / (laenge / 1000) if laenge > 0 else 0
                text = font_values.render(f"{kraft_pro_meter:.0f}N/m", True, COLORS['force_text'])
                
                mid_pos = (start_pos + end_pos) / 2
                if abs(end_amplitude) > 0:
                    kraft_unit = kraft_vektor_end / np.linalg.norm(kraft_vektor_end)
                    text_offset = kraft_unit * 15
                    text_pos = mid_pos + kraft_vektor_end + text_offset
                else:
                    text_pos = mid_pos + np.array([5, -5])
                text_rect = text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                surf.blit(text, text_rect)
            else:
                # Variable load
                start_kraft_pro_meter = abs(start_amplitude) / (laenge / 1000) if laenge > 0 else 0
                end_kraft_pro_meter = abs(end_amplitude) / (laenge / 1000) if laenge > 0 else 0
                
                if abs(start_amplitude) > 1e-6:
                    text_start = font_values.render(f"{start_kraft_pro_meter:.0f}N/m", True, COLORS['force_text'])
                    if abs(start_amplitude) > 0:
                        kraft_unit = kraft_vektor_start / np.linalg.norm(kraft_vektor_start)
                        text_offset = kraft_unit * 15
                        text_pos = start_pos + kraft_vektor_start + text_offset
                    else:
                        text_pos = start_pos + np.array([5, -5])
                    text_rect = text_start.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                    surf.blit(text_start, text_rect)
                
                if abs(end_amplitude) > 1e-6:
                    text_end = font_values.render(f"{end_kraft_pro_meter:.0f}N/m", True, COLORS['force_text'])
                    if abs(end_amplitude) > 0:
                        kraft_unit = kraft_vektor_end / np.linalg.norm(kraft_vektor_end)
                        text_offset = kraft_unit * 15
                        text_pos = end_pos + kraft_vektor_end + text_offset
                    else:
                        text_pos = end_pos + np.array([5, -5])
                    text_rect = text_end.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                    surf.blit(text_end, text_rect)
            
        # Draw support reactions
        ist_bestimmt, _ = self.pruefe_statische_bestimmtheit()
        if ist_bestimmt and len(self.lager) > 0:
            lager_reaktionen = self.berechne_lagerreaktionen()
            for lager_pos, (fx, fz, m_lager) in lager_reaktionen.items():
                if lager_pos == "start":
                    pos = self.start
                elif lager_pos == "end":
                    pos = self.end
                else:
                    continue
                    
                # Vertical reaction
                if abs(fz) > 0.1:
                    kraft_vektor = self.e_z * fz * 0.5
                    tip = pos + kraft_vektor
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, 2)
                    
                    arrow_length = 6
                    if abs(fz) > 0:
                        fz_unit = kraft_vektor / np.linalg.norm(kraft_vektor)
                        left_wing = tip - arrow_length * (fz_unit * np.cos(0.3) + 
                                   np.array([-fz_unit[1], fz_unit[0]]) * np.sin(0.3))
                        right_wing = tip - arrow_length * (fz_unit * np.cos(0.3) - 
                                    np.array([-fz_unit[1], fz_unit[0]]) * np.sin(0.3))
                        arrow_points = [tip, left_wing, right_wing]
                        pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)
                    
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fz:.0f}N", True, COLORS['reaction_text'])
                    text_pos = tip + np.array([5, -10])
                    surf.blit(text, text_pos.astype(int))
                    
                # Horizontal reaction
                if abs(fx) > 0.1:
                    kraft_vektor = self.e_x * fx * 0.5
                    tip = pos + kraft_vektor
                    pygame.draw.line(surf, COLORS['reaction'], pos, tip, 2)
                    
                    if abs(fx) > 0:
                        fx_unit = kraft_vektor / np.linalg.norm(kraft_vektor)
                        left_wing = tip - 6 * (fx_unit * np.cos(0.3) + 
                                   np.array([-fx_unit[1], fx_unit[0]]) * np.sin(0.3))
                        right_wing = tip - 6 * (fx_unit * np.cos(0.3) - 
                                    np.array([-fx_unit[1], fx_unit[0]]) * np.sin(0.3))
                        arrow_points = [tip, left_wing, right_wing]
                        pygame.draw.polygon(surf, COLORS['reaction'], arrow_points)
                    
                    font_reactions = get_font('reactions')
                    text = font_reactions.render(f"{fx:.0f}N", True, COLORS['force_text'])
                    text_pos = tip + np.array([5, 5])
                    surf.blit(text, text_pos.astype(int))
                    
        # Draw supports last to overlay everything
        for lager_pos, lager_typ in self.lager.items():
            if lager_pos == "start":
                self.draw_lager(surf, self.start, lager_typ)
            elif lager_pos == "end":
                self.draw_lager(surf, self.end, lager_typ)

    def draw_verlauf(self, surf, scale_factor=0.01):
        """Draw internal force diagrams"""
        if self.L == 0:
            return
        
        # Check if beam can be analyzed
        can_analyze = False
        ist_bestimmt = False
        
        if self.connected_structure and self.internal_forces_solved:
            # Part of solved connected structure
            can_analyze = True
            ist_bestimmt = True  # Connected structure is solved, so it's determinate
        else:
            # Individual beam analysis
            ist_bestimmt, status_text = self.pruefe_statische_bestimmtheit()
            can_analyze = ist_bestimmt
            
        if not can_analyze:
            return
        
        segments = self.get_segments()
        pts_N, pts_Q, pts_M = [], [], []
        beam_line_points = []
        
        # Generate points for each segment
        for i in range(len(segments) - 1):
            x_start = segments[i]
            x_end = segments[i + 1]
            
            num_points = max(5, int((x_end - x_start) / self.L * 100))
            
            for j in range(num_points + 1):
                if j == num_points and i < len(segments) - 2:
                    continue
                    
                t = j / num_points if num_points > 0 else 0
                x = x_start + t * (x_end - x_start)
                w = self.world_point(x)
                N, Q, M = self.schnittgroessen(x)
                
                beam_line_points.append(w)
                
                pts_N.append(w + self.e_z * N * scale_factor)
                pts_Q.append(w + self.e_z * Q * scale_factor)
                pts_M.append(w + self.e_z * M * scale_factor * 0.01)
        
        # Check for non-zero values
        has_N_values = any(abs(N_pt[1] - beam_pt[1]) > 0.1 for N_pt, beam_pt in zip(pts_N, beam_line_points))
        has_Q_values = any(abs(Q_pt[1] - beam_pt[1]) > 0.1 for Q_pt, beam_pt in zip(pts_Q, beam_line_points))
        has_M_values = any(abs(M_pt[1] - beam_pt[1]) > 0.1 for M_pt, beam_pt in zip(pts_M, beam_line_points))
        
        # Draw filled areas
        if len(pts_N) > 1 and len(beam_line_points) > 1 and has_N_values:
            n_polygon = pts_N + list(reversed(beam_line_points))
            draw_transparent_polygon(surf, COLORS['N'], n_polygon, 70)
            
        if len(pts_Q) > 1 and len(beam_line_points) > 1 and has_Q_values:
            q_polygon = pts_Q + list(reversed(beam_line_points))
            draw_transparent_polygon(surf, COLORS['Q'], q_polygon, 70)
            
        if len(pts_M) > 1 and len(beam_line_points) > 1 and has_M_values:
            m_polygon = pts_M + list(reversed(beam_line_points))
            draw_transparent_polygon(surf, COLORS['M'], m_polygon, 70)
        
        # Draw lines
        if len(pts_N) > 1 and has_N_values:
            pygame.draw.lines(surf, COLORS['N'], False, pts_N, 2)
        if len(pts_Q) > 1 and has_Q_values:
            pygame.draw.lines(surf, COLORS['Q'], False, pts_Q, 2)
        if len(pts_M) > 1 and has_M_values:
            pygame.draw.lines(surf, COLORS['M'], False, pts_M, 2)
            
        # Draw labels
        font_legend = get_font('legend')
        
        if ist_bestimmt:
            if len(pts_N) > 5 and has_N_values:
                mid_idx = len(pts_N) // 2
                n_pos = pts_N[mid_idx]
                beam_pos = beam_line_points[mid_idx]
                
                if n_pos[1] < beam_pos[1]:
                    text_pos = n_pos + np.array([0, -20])
                else:
                    text_pos = n_pos + np.array([0, 10])
                    
                n_text = font_legend.render("N(x)", True, COLORS['N'])
                surf.blit(n_text, text_pos.astype(int))
            
            if len(pts_Q) > 5 and has_Q_values:
                mid_idx = len(pts_Q) // 2
                q_pos = pts_Q[mid_idx]
                beam_pos = beam_line_points[mid_idx]
                
                if q_pos[1] < beam_pos[1]:
                    text_pos = q_pos + np.array([0, -20])
                else:
                    text_pos = q_pos + np.array([0, 10])
                    
                q_text = font_legend.render("Q(x)", True, COLORS['Q'])
                surf.blit(q_text, text_pos.astype(int))
            
            if len(pts_M) > 5 and has_M_values:
                mid_idx = len(pts_M) // 2
                m_pos = pts_M[mid_idx]
                beam_pos = beam_line_points[mid_idx]
                
                if m_pos[1] < beam_pos[1]:
                    text_pos = m_pos + np.array([0, -20])
                else:
                    text_pos = m_pos + np.array([0, 10])
                    
                m_text = font_legend.render("M(x)", True, COLORS['M'])
                surf.blit(m_text, text_pos.astype(int))

    def draw_basic(self, surf):
        """Basic drawing method for preview/compatibility"""
        self.draw(surf)

def draw_grid(surface):
    """Draw grid with small crosses at snapping points"""
    w, h = surface.get_size()
    cross_size = 3
    grid_color = COLORS['grid']
    
    for x in range(0, w, GRID_SIZE):
        for y in range(0, h, GRID_SIZE):
            pygame.draw.line(surface, grid_color, 
                           (x - cross_size, y), (x + cross_size, y), 1)
            pygame.draw.line(surface, grid_color, 
                           (x, y - cross_size), (x, y + cross_size), 1)

def draw_slider(surf, x, y, width, value, min_val, max_val, label):
    """Draw a slider"""
    slider_bg_color = (45, 55, 75)
    
    pygame.draw.rect(surf, slider_bg_color, (x, y, width, 20), 0)
    
    dimmer_blue = (90, 130, 180)
    pygame.draw.rect(surf, dimmer_blue, (x, y, width, 20), 2)
    
    slider_pos = x + (value - min_val) / (max_val - min_val) * width
    
    pygame.draw.circle(surf, slider_bg_color, (int(slider_pos), y + 10), 8)
    pygame.draw.circle(surf, dimmer_blue, (int(slider_pos), y + 10), 6)
    
    font_slider = get_font('slider')
    label_text = font_slider.render(f"{label}: {value:.2f}", True, COLORS['ui_text'])
    text_rect = label_text.get_rect()
    text_x = x + (width - text_rect.width) // 2
    surf.blit(label_text, (text_x, y + 25))
    
    return (x, y, width, 20)

def handle_slider_click(mouse_pos, slider_rect, min_val, max_val):
    """Handle slider clicks"""
    x, y, width, height = slider_rect
    extended_y = y - 10
    extended_height = height + 20
    
    if x <= mouse_pos[0] <= x + width and extended_y <= mouse_pos[1] <= extended_y + extended_height:
        relative_pos = (mouse_pos[0] - x) / width
        new_value = min_val + relative_pos * (max_val - min_val)
        return max(min_val, min(max_val, new_value))
    return None

# Initialize
structure = ConnectedBeamStructure()
mode = "idle"
clicks = []
temp_beam = None
scale_factor = 0.5
slider_dragging = False
show_debug = False
selected_beam = None

running = True
while running:
    screen.fill(COLORS['bg'])
    draw_grid(screen)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = snap(pygame.mouse.get_pos())
            
            # Check slider interaction first
            if len(structure.beams) > 0:
                any_statically_determinate = any(beam.pruefe_statische_bestimmtheit()[0] for beam in structure.beams)
                if any_statically_determinate:
                    slider_rect = (screen.get_width() - 220, 10, 200, 20)
                    new_scale = handle_slider_click(pygame.mouse.get_pos(), slider_rect, 0.01, 1.0)
                    if new_scale is not None:
                        scale_factor = new_scale
                        slider_dragging = True
                        continue

            if mode == "idle":
                clicks = [pos]
                mode = "beam_create"
            elif mode == "beam_create":
                clicks.append(pos)
                if np.linalg.norm(clicks[1] - clicks[0]) > 5:
                    new_beam = structure.add_beam(clicks[0], clicks[1])
                    if new_beam is None and len(structure.beams) > 0:
                        # Beam was rejected - show error message briefly
                        print("Beam must connect to existing structure!")
                    temp_beam = None
                    mode = "idle"
                    clicks = []
            elif mode == "punktlast1":
                # Find beam to add load to
                selected_beam = structure.find_beam_at_point(pos)
                if selected_beam:
                    snapped_pos = selected_beam.snap_to_beam(pos)
                    clicks = [snapped_pos]
                    mode = "punktlast2"
            elif mode == "punktlast2":
                if selected_beam:
                    snapped_pos = selected_beam.snap_to_beam(clicks[0])
                    selected_beam.add_punktlast(snapped_pos, pos - snapped_pos)
                mode = "punktlast1"
                clicks = []
            elif mode == "linelast1":
                selected_beam = structure.find_beam_at_point(pos)
                if selected_beam:
                    snapped_pos = selected_beam.snap_to_beam(pos)
                    clicks = [snapped_pos]
                    mode = "linelast2"
            elif mode == "linelast2":
                if selected_beam:
                    vec_to_click = pos - selected_beam.start
                    projection_length = np.dot(vec_to_click, selected_beam.e_x)
                    projection_length = max(0, min(selected_beam.L, projection_length))
                    snapped_pos = selected_beam.start + projection_length * selected_beam.e_x
                    clicks.append(snapped_pos)
                    mode = "linelast3"
            elif mode == "linelast3":
                if selected_beam:
                    mid = 0.5 * (clicks[0] + clicks[1])
                    richtung = pos - mid
                    selected_beam.add_streckenlast(clicks[0], clicks[1], richtung, richtung)
                mode = "linelast1"
                clicks = []
            elif mode == "streckenlast1":
                selected_beam = structure.find_beam_at_point(pos)
                if selected_beam:
                    snapped_pos = selected_beam.snap_to_beam(pos)
                    clicks = [snapped_pos]
                    mode = "streckenlast2"
            elif mode == "streckenlast2":
                if selected_beam:
                    vec_to_click = pos - selected_beam.start
                    projection_length = np.dot(vec_to_click, selected_beam.e_x)
                    projection_length = max(0, min(selected_beam.L, projection_length))
                    snapped_pos = selected_beam.start + projection_length * selected_beam.e_x
                    clicks.append(snapped_pos)
                    mode = "streckenlast3"
            elif mode == "streckenlast3":
                if selected_beam:
                    richtung = pos - 0.5 * (clicks[0] + clicks[1])
                    clicks.append(richtung)
                    mode = "streckenlast4"
            elif mode == "streckenlast4":
                if selected_beam:
                    end_richtung = clicks[2]
                    start_richtung = pos - 0.5 * (clicks[0] + clicks[1])
                    selected_beam.add_streckenlast(clicks[0], clicks[1], end_richtung, start_richtung)
                mode = "streckenlast1"
                clicks = []
            elif mode == "lager":
                selected_beam = structure.find_beam_at_point(pos)
                if selected_beam:
                    selected_beam.add_lager(pos)
                clicks = []

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:  # Right click
            pos = snap(pygame.mouse.get_pos())
            structure.toggle_joint_type(pos)
            
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            slider_dragging = False
            
        elif event.type == pygame.MOUSEMOTION and slider_dragging:
            if len(structure.beams) > 0:
                any_statically_determinate = any(beam.pruefe_statische_bestimmtheit()[0] for beam in structure.beams)
                if any_statically_determinate:
                    slider_rect = (screen.get_width() - 220, 10, 200, 20)
                    new_scale = handle_slider_click(pygame.mouse.get_pos(), slider_rect, 0.01, 1.0)
                    if new_scale is not None:
                        scale_factor = new_scale
                    
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                mode = "idle"
                clicks = []
                temp_beam = None
                selected_beam = None
            elif event.key == pygame.K_b:
                mode = "idle"
                clicks = []
                temp_beam = None
                selected_beam = None
            elif event.key == pygame.K_p:
                if len(structure.beams) > 0:
                    mode = "punktlast1"
                    clicks = []
                    selected_beam = None
            elif event.key == pygame.K_l:
                if len(structure.beams) > 0:
                    mode = "linelast1"
                    clicks = []
                    selected_beam = None
            elif event.key == pygame.K_t:
                if len(structure.beams) > 0:
                    mode = "streckenlast1"
                    clicks = []
                    selected_beam = None
            elif event.key == pygame.K_s:
                if len(structure.beams) > 0:
                    mode = "lager"
                    clicks = []
                    selected_beam = None
            elif event.key == pygame.K_c:
                structure = ConnectedBeamStructure()
                temp_beam = None
                mode = "idle"
                clicks = []
                selected_beam = None
            elif event.key == pygame.K_d:
                show_debug = not show_debug
    
    # Draw structure (this includes basic beam rectangles and joint symbols)
    structure.draw(screen)
    
    # Solve connected structure before drawing internal forces
    if len(structure.beams) > 0:
        solved, status = structure.solve_connected_structure()
    
    # Draw continuous internal force diagrams across the entire structure
    if len(structure.beams) > 0:
        structure.draw_continuous_internal_forces(screen, scale_factor)
    
    # Draw individual beam details (loads, forces, supports) without overwriting joints
    for beam in structure.beams:
        # Draw coordinate system and beam ID (already done in draw_basic, skip here)
        
        # Draw local coordinate system at start
        x_axis_end = beam.start + beam.e_x * 40
        z_axis_end = beam.start + beam.e_z * 40
        
        pygame.draw.line(screen, COLORS['x_axis'], beam.start, x_axis_end, 2)
        pygame.draw.line(screen, COLORS['z_axis'], beam.start, z_axis_end, 2)
        
        font_axis = get_font('axis')
        x_text = font_axis.render("x", True, COLORS['x_axis'])
        z_text = font_axis.render("z", True, COLORS['z_axis'])
        screen.blit(x_text, (x_axis_end + np.array([5, -10])).astype(int))
        screen.blit(z_text, (z_axis_end + np.array([5, -10])).astype(int))
        
        # Draw loads and reactions (everything except the basic beam shape)
        beam.draw_loads_and_reactions(screen)
        
        if show_debug:
            # Add debug info for connected structure
            if len(structure.beams) > 0:
                is_determinate, structure_status = structure.check_overall_static_determinacy()
                font_debug = get_font('debug')
                structure_debug_text = font_debug.render(f"Structure: {structure_status}", True, COLORS['force_text'])
                screen.blit(structure_debug_text, (10, 10))
                
            # Add debug info for each beam
            for i, beam in enumerate(structure.beams):
                ist_bestimmt, status_text = beam.pruefe_statische_bestimmtheit()
                font_debug = get_font('debug')
                debug_text = font_debug.render(f"B{beam.beam_id}: {status_text}", True, COLORS['force_text'])
                screen.blit(debug_text, (10, 30 + i * 40))
                
                # Show joint forces
                if beam.joint_forces_start is not None:
                    N, Q, M = beam.joint_forces_start
                    joint_text = font_debug.render(f"  Joint Start: N={N:.1f} Q={Q:.1f} M={M:.1f}", True, COLORS['force_text'])
                    screen.blit(joint_text, (10, 45 + i * 40))
                if beam.joint_forces_end is not None:
                    N, Q, M = beam.joint_forces_end
                    joint_text = font_debug.render(f"  Joint End: N={N:.1f} Q={Q:.1f} M={M:.1f}", True, COLORS['force_text'])
                    screen.blit(joint_text, (10, 60 + i * 40))
    
    # Draw preview
    mpos = snap(pygame.mouse.get_pos())
    if mode == "beam_create" and len(clicks) == 1:
        if np.linalg.norm(mpos - clicks[0]) > 5:
            temp_beam = Beam(clicks[0], mpos, 999)
            temp_beam.draw_basic(screen)
            
            # Show length
            beam_length_pixels = np.linalg.norm(mpos - clicks[0])
            beam_length_meters = beam_length_pixels / GRID_SIZE
            
            font_preview = get_font('preview')
            length_text = font_preview.render(f"{beam_length_meters:.1f}m", True, COLORS['force_text'])
            
            mid_point = (clicks[0] + mpos) / 2
            text_pos = mid_point + np.array([0, -30])
            text_rect = length_text.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            
            screen.blit(length_text, text_rect)
    
    elif mode == "punktlast2" and len(clicks) == 1 and selected_beam:
        # Point load preview
        tip = mpos
        start = clicks[0]
        
        kraft_vec = tip - start
        kraft_norm = np.linalg.norm(kraft_vec)
        
        if kraft_norm > 5:
            pygame.draw.line(screen, COLORS['force_preview'], start, tip, 2)
            
            arrow_length = 8
            arrow_angle = 0.3
            kraft_unit = kraft_vec / kraft_norm
            left_wing = tip - arrow_length * (kraft_unit * np.cos(arrow_angle) + 
                       np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
            right_wing = tip - arrow_length * (kraft_unit * np.cos(arrow_angle) - 
                        np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
            arrow_points = [tip, left_wing, right_wing]
            pygame.draw.polygon(screen, COLORS['force_preview'], arrow_points)
            
            if kraft_norm > 5:
                load_intensity = f"{kraft_norm:.0f}N"
                font_preview = get_font('preview')
                text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
                
                text_offset = kraft_unit * 15
                text_pos = tip + text_offset
                text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
                
                screen.blit(text_surface, text_rect)
        
    elif mode == "linelast2" and len(clicks) == 1 and selected_beam:
        # Line load start preview
        vec_to_mouse = mpos - selected_beam.start  
        projection_length = np.dot(vec_to_mouse, selected_beam.e_x)
        projection_length = max(0, min(selected_beam.L, projection_length))
        preview_end = selected_beam.start + projection_length * selected_beam.e_x
        pygame.draw.line(screen, COLORS['force_preview'], clicks[0], preview_end, 2)
        
    elif mode == "linelast3" and len(clicks) == 2 and selected_beam:
        # Uniform line load preview
        mid = 0.5 * (clicks[0] + clicks[1])
        mouse_vec = mpos - mid
        amplitude = np.dot(mouse_vec, selected_beam.e_z)
        kraft_vektor = selected_beam.e_z * amplitude
        
        rect_points = [clicks[0], clicks[1], clicks[1] + kraft_vektor, clicks[0] + kraft_vektor]
        draw_transparent_polygon(screen, COLORS['force_preview'], rect_points, 180)
        pygame.draw.polygon(screen, COLORS['force_display'], rect_points, 2)
        
        num_arrows = max(3, int(np.linalg.norm(clicks[1] - clicks[0]) / 30))
        
        for i in range(num_arrows):
            t = i / (num_arrows - 1) if num_arrows > 1 else 0
            arrow_start = clicks[0] + t * (clicks[1] - clicks[0])
            arrow_end = arrow_start + kraft_vektor
            
            if np.linalg.norm(kraft_vektor) > 1e-6:
                pygame.draw.line(screen, COLORS['force_preview'], arrow_start, arrow_end, 2)
                
                arrow_length = 8
                arrow_angle = 0.3
                kraft_norm = np.linalg.norm(kraft_vektor)
                if kraft_norm > 0:
                    kraft_unit = kraft_vektor / kraft_norm
                    left_wing = arrow_end - arrow_length * (kraft_unit * np.cos(arrow_angle) + 
                               np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                    right_wing = arrow_end - arrow_length * (kraft_unit * np.cos(arrow_angle) - 
                                np.array([-kraft_unit[1], kraft_unit[0]]) * np.sin(arrow_angle))
                    arrow_points = [arrow_end, left_wing, right_wing]
                    pygame.draw.polygon(screen, COLORS['force_preview'], arrow_points)
        
        if num_arrows > 0 and np.linalg.norm(kraft_vektor) > 5:
            line_length = np.linalg.norm(clicks[1] - clicks[0])
            load_intensity_per_meter = np.linalg.norm(kraft_vektor) / line_length * 1000 if line_length > 0 else 0
            load_intensity = f"{load_intensity_per_meter:.0f}N/m (uniform)"
            font_preview = get_font('preview')
            text_surface = font_preview.render(load_intensity, True, COLORS['force_text'])
            
            kraft_norm = np.linalg.norm(kraft_vektor)
            kraft_unit = kraft_vektor / kraft_norm
            mid_point = (clicks[0] + clicks[1]) / 2
            arrow_end = mid_point + kraft_vektor
            text_offset = kraft_unit * 15
            text_pos = arrow_end + text_offset
            text_rect = text_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))
            
            screen.blit(text_surface, text_rect)

    # Draw slider if any beam is statically determinate
    if len(structure.beams) > 0:
        any_statically_determinate = any(beam.pruefe_statische_bestimmtheit()[0] for beam in structure.beams)
        if any_statically_determinate:
            slider_rect = draw_slider(screen, screen.get_width() - 220, 10, 200, scale_factor, 0.01, 1.0, "Graph Scale")

    # UI
    font_ui = get_font('ui')
    
    shortcuts_line1 = "B: Beam | P: Point Load | L: Line Load | S: Support"
    shortcuts_line2 = "T: Trapezoidal Load | C: Clear | D: Debug | ESC: Cancel"
    shortcuts_text1 = font_ui.render(shortcuts_line1, True, COLORS['ui_text'])
    shortcuts_text2 = font_ui.render(shortcuts_line2, True, COLORS['ui_text'])
    screen.blit(shortcuts_text1, (10, 10))
    screen.blit(shortcuts_text2, (10, 35))
    
    # Show connected joints info and structural status
    if len(structure.beams) > 0:
        connected_joints = len([j for j in structure.joints.values() if len(j.get('beams', [])) > 1])
        joint_info = f"Beams: {len(structure.beams)} | Connected Joints: {connected_joints}"
        joint_text = font_ui.render(joint_info, True, COLORS['status_ok'])
        screen.blit(joint_text, (10, screen.get_height() - 75))
        
        # Show overall structural determinacy
        is_determinate, structure_status = structure.check_overall_static_determinacy()
        status_color = COLORS['status_ok'] if is_determinate else COLORS['status_error']
        status_text = font_ui.render(f"Structure: {structure_status}", True, status_color)
        screen.blit(status_text, (10, screen.get_height() - 55))
    
    # Status display
    if mode != "idle":
        if mode == "beam_create":
            msg = "Beam: Click second point"
        elif mode == "punktlast1":
            msg = "Point Load: Click on beam to add load"
        elif mode == "punktlast2":
            msg = "Point Load: Set force direction and magnitude"
        elif mode == "linelast1":
            msg = "Uniform Line Load: Click on beam start point"
        elif mode == "linelast2":
            msg = "Uniform Line Load: Click end point"
        elif mode == "linelast3":
            msg = "Uniform Line Load: Set uniform force magnitude (perpendicular)"
        elif mode == "streckenlast1":
            msg = "Trapezoidal Load: Click on beam start point"
        elif mode == "streckenlast2":
            msg = "Trapezoidal Load: Click end point"
        elif mode == "streckenlast3":
            msg = "Trapezoidal Load: Set force magnitude at START (perpendicular)"
        elif mode == "streckenlast4":
            msg = "Trapezoidal Load: Set force magnitude at END (variable load)"
        elif mode == "lager":
            msg = "Support: Click beam end (multiple clicks to change type)"
        else:
            msg = f"Mode: {mode}"
        
        status_text = font_ui.render(msg, True, COLORS['ui_text'])
        screen.blit(status_text, (10, screen.get_height() - 30))
    else:
        # Show joint instructions
        msg = "Right-click joints to toggle: Rigid Joint (R) → Pin (P) → Free (F) → Rigid Connection (continuous)"
        status_text = font_ui.render(msg, True, COLORS['ui_text'])
        screen.blit(status_text, (10, screen.get_height() - 30))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
