import pygame
import sys
import math
import numpy as np
from enum import Enum

# Klassen und Enums
class Modus(Enum):
    BALKEN_ZEICHNEN = 1
    LAGER_SETZEN = 2
    PUNKTLASTEN_SETZEN = 3
    LINIENLASTEN_SETZEN = 4
    BEARBEITEN = 5
    BERECHNUNG = 6

class LagerTyp(Enum):
    LOSLAGER = 1        # F_x = 0, M = 0 (1 Wertigkeit: F_z)
    FESTLAGER = 2       # M = 0 (2 Wertigkeit: F_x, F_z)
    PARALLELFUEHRUNG = 3 # F_z = 0 (2 Wertigkeit: F_x, M)
    SCHIEBEHUELSE = 4   # F_x = 0 (2 Wertigkeit: F_z, M)  
    EINSPANNUNG = 5     # (3 Wertigkeit: F_x, F_z, M)
    GELENK = 6          # M = 0 (nur zwischen Balken, 2 Wertigkeit)

class Balken:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.laenge = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        self.lager = []
        self.punktlasten = []
        self.linienlasten = []
        self.selected = False
        self.selected_lager = None
        self.selected_punktlast = None
        self.selected_linienlast = None
    
    def draw(self, screen):
        color = (200, 100, 255) if self.selected else (0, 0, 0)  # Lila wenn ausgewählt
        pygame.draw.line(screen, color, self.start, self.end, 6)  # Dickerer Balken (war 4)
        
        # Zeichne Lager
        for lager in self.lager:
            self.draw_lager(screen, lager)
        
        # Zeichne Punktlasten
        for last in self.punktlasten:
            self.draw_punktlast(screen, last)
            
        # Zeichne Linienlasten
        for last in self.linienlasten:
            self.draw_linienlast(screen, last)
    
    def draw_lager(self, screen, lager):
        x, y = lager['position']
        # Lila Farbe wenn ausgewählt
        selected = self.selected_lager == lager
        fill_color = (200, 100, 255) if selected else (100, 100, 100)
        border_color = (150, 0, 255) if selected else (0, 0, 0)
        
        if lager['typ'] == LagerTyp.FESTLAGER:
            # Festlager: Dreieck mit Schraffur unten
            points = [(x, y), (x-12, y+15), (x+12, y+15)]
            pygame.draw.polygon(screen, fill_color, points)
            pygame.draw.polygon(screen, border_color, points, 2)
            # Schraffur für Festlager
            for i in range(-10, 13, 4):
                pygame.draw.line(screen, border_color, (x+i, y+15), (x+i+3, y+20), 2)
                
        elif lager['typ'] == LagerTyp.LOSLAGER:
            # Loslager: Dreieck mit Rollen
            points = [(x, y), (x-12, y+15), (x+12, y+15)]
            pygame.draw.polygon(screen, fill_color, points)
            pygame.draw.polygon(screen, border_color, points, 2)
            # Rollen für Loslager
            pygame.draw.circle(screen, fill_color, (x-6, y+18), 3)
            pygame.draw.circle(screen, border_color, (x-6, y+18), 3, 1)
            pygame.draw.circle(screen, fill_color, (x+6, y+18), 3)
            pygame.draw.circle(screen, border_color, (x+6, y+18), 3, 1)
            # Grundlinie
            pygame.draw.line(screen, border_color, (x-15, y+21), (x+15, y+21), 2)
            
        elif lager['typ'] == LagerTyp.EINSPANNUNG:
            # Einspannung: Rechteck mit Schraffur
            pygame.draw.rect(screen, fill_color, (x-3, y-12, 6, 24))
            pygame.draw.rect(screen, border_color, (x-3, y-12, 6, 24), 2)
            # Schraffur für Einspannung (schräg)
            for i in range(-8, 12, 3):
                pygame.draw.line(screen, border_color, (x-3, y+i), (x-8, y+i-5), 2)
                
        elif lager['typ'] == LagerTyp.GELENK:
            # Gelenk: Kreis
            pygame.draw.circle(screen, fill_color, (x, y), 8)
            pygame.draw.circle(screen, border_color, (x, y), 8, 2)
    
    def draw_punktlast(self, screen, last):
        x, y = last['position']
        kraft_x = last.get('kraft_x', 0)  # Horizontale Komponente
        kraft_y = last.get('kraft_y', 0)  # Vertikale Komponente
        
        # Fallback für alte Punktlasten (nur kraft)
        if kraft_x == 0 and kraft_y == 0:
            kraft_y = last.get('kraft', 0)
        
        # Lila Farbe wenn ausgewählt
        selected = self.selected_punktlast == last
        color = (200, 100, 255) if selected else (255, 0, 0)
        
        # Kraftbetrag berechnen
        kraft_betrag = math.sqrt(kraft_x**2 + kraft_y**2)
        
        if kraft_betrag > 1:
            # Pfeil zeichnen (beliebige Richtung)
            end_x = x + kraft_x
            end_y = y + kraft_y
            pygame.draw.line(screen, color, (x, y), (end_x, end_y), 3)
            
            # Pfeilspitze berechnen
            if kraft_betrag > 0:
                # Richtungsvektor normalisieren
                dx_norm = kraft_x / kraft_betrag
                dy_norm = kraft_y / kraft_betrag
                
                # Pfeilspitze 10px vor dem Ende
                spitze_x = end_x - dx_norm * 10
                spitze_y = end_y - dy_norm * 10
                
                # Senkrechter Vektor für Pfeilflügel
                perp_x = -dy_norm * 5
                perp_y = dx_norm * 5
                
                # Pfeilspitze zeichnen
                pygame.draw.polygon(screen, color, [
                    (end_x, end_y),
                    (spitze_x + perp_x, spitze_y + perp_y),
                    (spitze_x - perp_x, spitze_y - perp_y)
                ])
            
            # Kraftwert an der Pfeilspitze anzeigen
            font = pygame.font.Font(None, 20)
            kraft_text = f"{int(kraft_betrag)}N"
            text_surf = font.render(kraft_text, True, color)
            # Position leicht versetzt von der Pfeilspitze
            text_x = end_x + 5
            text_y = end_y - 10
            screen.blit(text_surf, (text_x, text_y))
    
    def draw_linienlast(self, screen, last):
        """Zeichnet eine Linienlast senkrecht zum Balken"""
        start_pos = last['start']
        end_pos = last['end']
        kraft = last['kraft']
        
        # Lila Farbe wenn ausgewählt
        selected = self.selected_linienlast == last
        color = (200, 100, 255) if selected else ((255, 0, 0) if kraft < 0 else (0, 255, 0))
        
        # Balkenrichtung bestimmen
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge > 0:
            # Normierter Normalvektor (senkrecht zum Balken)
            normal_x = -dy / balken_laenge
            normal_y = dx / balken_laenge
            
            # Endpunkte der Kraftvektoren
            kraft_start_x = start_pos[0] + kraft * normal_x
            kraft_start_y = start_pos[1] + kraft * normal_y
            kraft_end_x = end_pos[0] + kraft * normal_x
            kraft_end_y = end_pos[1] + kraft * normal_y
            
            # Rechteck für Linienlast zeichnen - nur Umrandung, keine Füllung
            points = [
                start_pos,
                end_pos,
                (kraft_end_x, kraft_end_y),
                (kraft_start_x, kraft_start_y)
            ]
            pygame.draw.polygon(screen, color, points, 2)  # Nur Umrandung
            
            # Zusätzliche Linien für bessere Sichtbarkeit
            # Verbindungslinien zwischen Start- und Endpunkten
            pygame.draw.line(screen, color, start_pos, (kraft_start_x, kraft_start_y), 1)
            pygame.draw.line(screen, color, end_pos, (kraft_end_x, kraft_end_y), 1)
            
            # Pfeile entlang der Linienlast zeichnen
            linienlast_laenge = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            anzahl_pfeile = max(3, int(linienlast_laenge / 25))
            
            for i in range(anzahl_pfeile + 1):
                t = i / anzahl_pfeile if anzahl_pfeile > 0 else 0
                
                # Punkt auf der Balkenlinie
                pfeil_start_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                pfeil_start_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                
                # Endpunkt des Pfeils
                pfeil_end_x = pfeil_start_x + kraft * normal_x
                pfeil_end_y = pfeil_start_y + kraft * normal_y
                
                # Pfeil zeichnen
                pygame.draw.line(screen, color, (pfeil_start_x, pfeil_start_y), (pfeil_end_x, pfeil_end_y), 2)
                
                # Pfeilspitze - korrigierte Richtung
                if abs(kraft) > 5:
                    spitze_laenge = 8
                    # Pfeilspitze zeigt in Kraftrichtung
                    pfeil_richtung_x = kraft * normal_x
                    pfeil_richtung_y = kraft * normal_y
                    pfeil_laenge = math.sqrt(pfeil_richtung_x**2 + pfeil_richtung_y**2)
                    
                    if pfeil_laenge > 0:
                        # Normierte Richtung der Kraft
                        norm_richtung_x = pfeil_richtung_x / pfeil_laenge
                        norm_richtung_y = pfeil_richtung_y / pfeil_laenge
                        
                        # Senkrechte zur Pfeilrichtung
                        perp_x = -norm_richtung_y
                        perp_y = norm_richtung_x
                        
                        # Pfeilspitze berechnen
                        spitze_x1 = pfeil_end_x - spitze_laenge * norm_richtung_x + spitze_laenge * 0.3 * perp_x
                        spitze_y1 = pfeil_end_y - spitze_laenge * norm_richtung_y + spitze_laenge * 0.3 * perp_y
                        spitze_x2 = pfeil_end_x - spitze_laenge * norm_richtung_x - spitze_laenge * 0.3 * perp_x
                        spitze_y2 = pfeil_end_y - spitze_laenge * norm_richtung_y - spitze_laenge * 0.3 * perp_y
                        
                        pygame.draw.polygon(screen, color, [
                            (pfeil_end_x, pfeil_end_y),
                            (spitze_x1, spitze_y1),
                            (spitze_x2, spitze_y2)
                        ])
            
            # Kraftwert in der Mitte der Linienlast anzeigen
            font = pygame.font.Font(None, 18)
            kraft_text = f"{abs(int(kraft))}N/m"
            text_surf = font.render(kraft_text, True, color)
            # Position in der Mitte der Linienlast, etwas vom Balken weg
            mid_x = (start_pos[0] + end_pos[0]) // 2 + int(kraft * normal_x * 0.7)
            mid_y = (start_pos[1] + end_pos[1]) // 2 + int(kraft * normal_y * 0.7)
            screen.blit(text_surf, (mid_x, mid_y))
    
    def point_on_beam(self, point, tolerance=25):
        """Prüft ob ein Punkt nahe dem Balken liegt - erweiterte Toleranz für bessere Usability"""
        x1, y1 = self.start
        x2, y2 = self.end
        px, py = point
        
        # Balken-Vektor
        dx = x2 - x1
        dy = y2 - y1
        balken_laenge_squared = dx*dx + dy*dy
        
        if balken_laenge_squared == 0:  # Balken hat keine Länge
            return abs(px - x1) <= tolerance and abs(py - y1) <= tolerance
        
        # Projektion des Punktes auf die Balkenlinie (Parameter t)
        t = ((px - x1) * dx + (py - y1) * dy) / balken_laenge_squared
        
        # Nächster Punkt auf der erweiterten Linie (auch außerhalb des Balkens)
        naechster_x = x1 + t * dx
        naechster_y = y1 + t * dy
        
        # Distanz vom Punkt zur Balkenlinie
        distance = math.sqrt((px - naechster_x)**2 + (py - naechster_y)**2)
        
        # Prüfen ob Punkt innerhalb der Toleranz liegt
        # UND ob die Projektion im erweiterten Balkenbereich liegt (erweitert um Toleranz)
        balken_laenge = math.sqrt(balken_laenge_squared)
        t_tolerance = tolerance / balken_laenge if balken_laenge > 0 else 0
        
        return distance <= tolerance and (-t_tolerance <= t <= 1 + t_tolerance)
    
    def get_clicked_lager(self, pos, tolerance=30):
        """Gibt das angeklickte Lager zurück - erhöhte Toleranz für bessere Usability"""
        for lager in self.lager:
            lager_x, lager_y = lager['position']
            if abs(pos[0] - lager_x) <= tolerance and abs(pos[1] - lager_y) <= tolerance:
                return lager
        return None
    
    def get_clicked_punktlast(self, pos, tolerance=35):
        """Gibt die angeklickte Punktlast zurück - erhöhte Toleranz für bessere Usability"""
        for last in self.punktlasten:
            last_x, last_y = last['position']
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            
            # Fallback für alte Punktlasten (nur kraft)
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            # Endpunkt des Kraftpfeils
            end_x = last_x + kraft_x
            end_y = last_y + kraft_y
            
            # Prüfe Klick auf Pfeil-Linie oder Startpunkt
            # 1. Klick auf Startpunkt
            if abs(pos[0] - last_x) <= tolerance and abs(pos[1] - last_y) <= tolerance:
                return last
            
            # 2. Klick auf Pfeil-Linie (vereinfacht)
            kraft_betrag = math.sqrt(kraft_x**2 + kraft_y**2)
            if kraft_betrag > 0:
                # Prüfe ob Punkt nahe der Pfeillinie liegt
                # Verwende vereinfachte Rechteck-Prüfung um die Linie
                min_x = min(last_x, end_x) - tolerance
                max_x = max(last_x, end_x) + tolerance
                min_y = min(last_y, end_y) - tolerance
                max_y = max(last_y, end_y) + tolerance
                
                if min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y:
                    return last
        
        return None
    
    def get_clicked_linienlast(self, pos, tolerance=30):
        """Gibt die angeklickte Linienlast zurück - erhöhte Toleranz für bessere Usability"""
        for last in self.linienlasten:
            start_x, start_y = last['start']
            end_x, end_y = last['end']
            kraft = last['kraft']
            
            # Balkenrichtung bestimmen für bessere Auswahl
            dx = self.end[0] - self.start[0]
            dy = self.end[1] - self.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge > 0:
                # Normierter Normalvektor (senkrecht zum Balken)
                normal_x = -dy / balken_laenge
                normal_y = dx / balken_laenge
                
                # Endpunkte der Kraftvektoren
                kraft_start_x = start_x + kraft * normal_x
                kraft_start_y = start_y + kraft * normal_y
                kraft_end_x = end_x + kraft * normal_x
                kraft_end_y = end_y + kraft * normal_y
                
                # Erweiterte Rechteck-Prüfung für Linienlast
                min_x = min(start_x, end_x, kraft_start_x, kraft_end_x) - tolerance
                max_x = max(start_x, end_x, kraft_start_x, kraft_end_x) + tolerance
                min_y = min(start_y, end_y, kraft_start_y, kraft_end_y) - tolerance
                max_y = max(start_y, end_y, kraft_start_y, kraft_end_y) + tolerance
                
                if min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y:
                    return last
        
        return None
    
    def get_clicked_lager(self, pos, tolerance=20):
        """Gibt das angeklickte Lager zurück"""
        for lager in self.lager:
            lx, ly = lager['position']
            if abs(pos[0] - lx) <= tolerance and abs(pos[1] - ly) <= tolerance:
                return lager
        return None
    
    def get_clicked_punktlast(self, pos, tolerance=25):
        """Gibt die angeklickte Punktlast zurück"""
        for last in self.punktlasten:
            lx, ly = last['position']
            kraft_y = last.get('kraft', 0)
            
            # Prüfe vertikale Kraft
            if kraft_y != 0:
                end_y = ly + abs(kraft_y) if kraft_y < 0 else ly - abs(kraft_y)
                min_y, max_y = min(ly, end_y), max(ly, end_y)
                
                if abs(pos[0] - lx) <= tolerance and min_y - tolerance <= pos[1] <= max_y + tolerance:
                    return last
            
            # Prüfe Position ohne Kraft (falls Kraft 0 ist)
            if kraft_y == 0:
                if abs(pos[0] - lx) <= tolerance and abs(pos[1] - ly) <= tolerance:
                    return last
        
        return None
    
    def get_clicked_linienlast(self, pos, tolerance=20):
        """Gibt die angeklickte Linienlast zurück"""
        for last in self.linienlasten:
            start_x, start_y = last['start']
            end_x, end_y = last['end']
            kraft = last['kraft']
            
            # Prüfe ob Punkt im erweiterten Linienlast-Rechteck liegt
            kraft_end_y = start_y + abs(kraft) if kraft < 0 else start_y - abs(kraft)
            
            min_x, max_x = min(start_x, end_x) - tolerance, max(start_x, end_x) + tolerance
            min_y, max_y = min(start_y, kraft_end_y) - tolerance, max(start_y, kraft_end_y) + tolerance
            
            if min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y:
                return last
        return None

class SchnittkraftTool:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption("Schnittkraft Tool – Baustatik Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Grid-System
        self.grid_size = 20
        self.show_grid = True
        
        self.balken_liste = []
        self.startpunkt = None
        self.selected_balken = None
        self.modus = Modus.BALKEN_ZEICHNEN
        
        # Punktlast ziehen
        self.dragging_punktlast = False
        self.punktlast_start = None
        self.punktlast_balken = None
        self.punktlast_horizontal = False  # Flag für horizontale Kräfte
        
        # Linienlast 3-Klick-System
        self.linienlast_state = 0  # 0: bereit, 1: start gesetzt, 2: end gesetzt, 3: kraft setzen
        self.linienlast_start = None
        self.linienlast_end = None
        self.linienlast_balken = None
        
        # GUI Elemente - Buttons werden dynamisch positioniert
        self.buttons = {}
        self.init_buttons()
        
        # Schieberegler für Schnittkraft-Skalierung (zweite Reihe)
        self.scale_slider = {
            'rect': pygame.Rect(10, 60, 180, 30),
            'handle_rect': pygame.Rect(90, 65, 20, 20),
            'dragging': False,
            'min_scale': 0.1,
            'max_scale': 10.0,
            'current_scale': 3.0
        }
    
    def init_buttons(self):
        """Initialisiert die Buttons mit automatischer Größenanpassung"""
        button_texts = [
            ('balken', 'Balken'),
            ('lager', 'Lager'),
            ('punktlast', 'Punktlast'),
            ('linienlast', 'Linienlast'),
            ('bearbeiten', 'Bearbeiten'),
            ('grid', 'Grid'),
            ('clear', 'Clear All')
        ]
        
        current_x = 10
        button_height = 40
        margin = 20  # Margin links/rechts vom Text
        
        for key, text in button_texts:
            # Textbreite messen
            text_surface = self.font.render(text, True, (0, 0, 0))
            text_width = text_surface.get_width()
            
            # Button-Breite = Textbreite + Margin
            button_width = text_width + margin
            
            # Button erstellen
            self.buttons[key] = {
                'rect': pygame.Rect(current_x, 10, button_width, button_height),
                'text': text
            }
            
            # Nächste x-Position
            current_x += button_width + 10  # 10px Abstand zwischen Buttons
    
    def snap_to_grid(self, pos):
        """Snappt Position zum Grid"""
        if self.show_grid:
            x, y = pos
            grid_x = round(x / self.grid_size) * self.grid_size
            grid_y = round(y / self.grid_size) * self.grid_size
            return (grid_x, grid_y)
        return pos
    
    def draw_grid(self):
        """Zeichnet das Grid"""
        if not self.show_grid:
            return
            
        width, height = self.screen.get_size()
        
        # Vertikale Linien
        for x in range(0, width, self.grid_size):
            pygame.draw.line(self.screen, (220, 220, 220), (x, 100), (x, height), 1)
        
        # Horizontale Linien
        for y in range(100, height, self.grid_size):
            pygame.draw.line(self.screen, (220, 220, 220), (0, y), (width, y), 1)
    
    def draw_scale_slider(self):
        """Zeichnet den Schieberegler für die Schnittkraft-Skalierung"""
        # Slider-Hintergrund
        pygame.draw.rect(self.screen, (200, 200, 200), self.scale_slider['rect'])
        pygame.draw.rect(self.screen, (100, 100, 100), self.scale_slider['rect'], 2)
        
        # Slider-Handle
        pygame.draw.rect(self.screen, (80, 80, 80), self.scale_slider['handle_rect'])
        pygame.draw.rect(self.screen, (50, 50, 50), self.scale_slider['handle_rect'], 2)
        
        # Beschriftung über dem Slider
        scale_text = f"Skalierung: {self.scale_slider['current_scale']:.1f}x"
        text_surf = self.font.render(scale_text, True, (0, 0, 0))
        self.screen.blit(text_surf, (self.scale_slider['rect'].x, self.scale_slider['rect'].y - 20))
    
    def draw_gui(self):
        # Buttons zeichnen
        for key, button in self.buttons.items():
            # Button Farbe je nach Modus
            if (key == 'balken' and self.modus == Modus.BALKEN_ZEICHNEN) or \
               (key == 'lager' and self.modus == Modus.LAGER_SETZEN) or \
               (key == 'punktlast' and self.modus == Modus.PUNKTLASTEN_SETZEN) or \
               (key == 'linienlast' and self.modus == Modus.LINIENLASTEN_SETZEN) or \
               (key == 'bearbeiten' and self.modus == Modus.BEARBEITEN) or \
               (key == 'grid' and self.show_grid):
                color = (100, 150, 255)
            else:
                color = (200, 200, 200)
            
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, (0, 0, 0), button['rect'], 2)
            
            # Button Text
            text_surf = self.font.render(button['text'], True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=button['rect'].center)
            self.screen.blit(text_surf, text_rect)
        
        # Modus-Info entfernt - Button zeigt aktiven Modus
        
        # Zusätzliche Info für verschiedene Modi
        if self.modus == Modus.LAGER_SETZEN:
            info = "Klick auf Balkenende: Lager setzen | Klick auf Lager: Typ wechseln | Gelenke nur an Verbindungen"
            text_surf = self.font.render(info, True, (0, 0, 255))
            self.screen.blit(text_surf, (10, 650))
        elif self.modus == Modus.PUNKTLASTEN_SETZEN:
            info = "Klick auf Balken + Ziehen für beliebige Kraftrichtungen - Länge = Kraftgröße"
            text_surf = self.font.render(info, True, (0, 0, 255))
            self.screen.blit(text_surf, (10, 650))
        elif self.modus == Modus.LINIENLASTEN_SETZEN:
            if self.linienlast_state == 0:
                info = "1. Start der Linienlast auf Balken setzen"
            elif self.linienlast_state == 1:
                info = "2. Ende der Linienlast auf demselben Balken setzen"
            elif self.linienlast_state == 2:
                info = "3. Kraft in beliebige Richtung ziehen (senkrecht zum Balken wird projiziert)"
            
            text_surf = self.font.render(info, True, (0, 0, 255))
            self.screen.blit(text_surf, (10, 650))
        
        elif self.modus == Modus.BEARBEITEN:
            info = "Elemente anklicken zum Auswählen - [Entf] zum Löschen"
            text_surf = self.font.render(info, True, (0, 0, 255))
            self.screen.blit(text_surf, (10, 650))
        
        # Info für ausgewählten Balken (nur im Bearbeiten-Modus)
        if self.selected_balken and self.modus == Modus.BEARBEITEN:
            info = "Element ausgewählt - [Entf] zum Löschen"
            text_surf = self.font.render(info, True, (255, 0, 0))
            self.screen.blit(text_surf, (10, 620))
        
        # Status der automatischen Berechnung - korrekte statische Analyse
        bestimmt, status_text = self.ist_system_statisch_bestimmt()
        if bestimmt:
            status_color = (0, 150, 0)
        else:
            status_color = (150, 0, 0)
        
        text_surf = self.font.render(status_text, True, status_color)
        self.screen.blit(text_surf, (10, 590))
        
        # Schieberegler für Schnittkraft-Skalierung zeichnen
        self.draw_scale_slider()
    
    def handle_button_click(self, pos):
        for key, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
                if key == 'balken':
                    self.modus = Modus.BALKEN_ZEICHNEN
                    self.selected_balken = None
                    self.reset_linienlast_state()
                elif key == 'lager':
                    self.modus = Modus.LAGER_SETZEN
                    self.reset_linienlast_state()
                elif key == 'punktlast':
                    self.modus = Modus.PUNKTLASTEN_SETZEN
                    self.reset_linienlast_state()
                elif key == 'linienlast':
                    self.modus = Modus.LINIENLASTEN_SETZEN
                    self.reset_linienlast_state()
                elif key == 'bearbeiten':
                    self.modus = Modus.BEARBEITEN
                    self.reset_linienlast_state()
                elif key == 'grid':
                    self.show_grid = not self.show_grid
                elif key == 'clear':
                    self.clear_all_elements()
                return True
        return False
    
    def clear_all_elements(self):
        """Löscht alle Balken und Elemente"""
        self.balken_liste = []
        self.selected_balken = None
        self.reset_linienlast_state()
        # Auch alle Zieh-Stati zurücksetzen
        self.dragging_punktlast = False
        self.punktlast_start = None
        self.punktlast_balken = None
    
    def reset_linienlast_state(self):
        """Setzt den Linienlast-Zustand zurück"""
        self.linienlast_state = 0
        self.linienlast_start = None
        self.linienlast_end = None
        self.linienlast_balken = None
    
    def handle_balken_mode(self, pos):
        pos = self.snap_to_grid(pos)
        if not self.startpunkt:
            self.startpunkt = pos
        else:
            # Beliebige Balkenrichtung erlauben (auch diagonal)
            endpunkt = pos
            
            # Mindestlänge prüfen
            dx = endpunkt[0] - self.startpunkt[0]
            dy = endpunkt[1] - self.startpunkt[1]
            laenge = math.sqrt(dx**2 + dy**2)
            
            if laenge > 20:  # Mindestlänge 20px
                self.balken_liste.append(Balken(self.startpunkt, endpunkt))
            
            self.startpunkt = None
    
    def handle_lager_mode(self, pos):
        pos = self.snap_to_grid(pos)
        
        # Prüfe ob Position an einem Balkenende liegt
        clicked_balken = None
        end_position = None
        
        for balken in self.balken_liste:
            # Prüfe Start- und Endpunkt des Balkens
            if self.points_close(pos, balken.start, tolerance=15):
                clicked_balken = balken
                end_position = balken.start
                break
            elif self.points_close(pos, balken.end, tolerance=15):
                clicked_balken = balken
                end_position = balken.end
                break
        
        if clicked_balken and end_position:
            # Prüfe ob bereits ein Lager an dieser Position existiert
            existing_lager = self.find_lager_at_position(end_position)
            
            if existing_lager:
                # Lager zyklisch ändern oder entfernen
                self.cycle_lager_at_position(end_position)
            else:
                # Neues Lager hinzufügen - aber nur wenn erlaubt
                if self.can_place_lager_at_position(end_position):
                    clicked_balken.lager.append({
                        'position': end_position,
                        'typ': LagerTyp.LOSLAGER  # Startet mit Loslager
                    })
    
    def find_lager_at_position(self, position):
        """Findet ein Lager an der angegebenen Position (alle Balken)"""
        for balken in self.balken_liste:
            for lager in balken.lager:
                if self.points_close(lager['position'], position, tolerance=15):
                    return lager
        return None
    
    def can_place_lager_at_position(self, position):
        """Prüft ob ein Lager an der Position gesetzt werden kann"""
        # Prüfe ob Position ein freies Ende oder eine Balkenverbindung ist
        connected_balken = self.get_balken_at_position(position)
        
        if len(connected_balken) == 1:
            # Freies Ende - alle Lagertypen erlaubt
            return True
        elif len(connected_balken) > 1:
            # Balkenverbindung - nur wenn noch kein Lager existiert
            return self.find_lager_at_position(position) is None
        
        return False
    
    def cycle_lager_at_position(self, position):
        """Wechselt das Lager an der Position zyklisch durch alle Typen"""
        existing_lager = self.find_lager_at_position(position)
        if not existing_lager:
            return
        
        connected_balken = self.get_balken_at_position(position)
        is_connection = len(connected_balken) > 1
        is_free_end = len(connected_balken) == 1
        
        # Zyklische Reihenfolge der Lagertypen
        if existing_lager['typ'] == LagerTyp.LOSLAGER:
            existing_lager['typ'] = LagerTyp.FESTLAGER
        elif existing_lager['typ'] == LagerTyp.FESTLAGER:
            existing_lager['typ'] = LagerTyp.PARALLELFUEHRUNG
        elif existing_lager['typ'] == LagerTyp.PARALLELFUEHRUNG:
            existing_lager['typ'] = LagerTyp.SCHIEBEHUELSE
        elif existing_lager['typ'] == LagerTyp.SCHIEBEHUELSE:
            existing_lager['typ'] = LagerTyp.EINSPANNUNG
        elif existing_lager['typ'] == LagerTyp.EINSPANNUNG:
            # Gelenk nur an Verbindungen, nicht an freien Enden
            if is_connection:
                existing_lager['typ'] = LagerTyp.GELENK
            else:
                # An freien Enden: Lager entfernen (vollständiger Zyklus)
                for balken in self.balken_liste:
                    for lager in balken.lager[:]:
                        if (self.points_close(lager['position'], position, tolerance=15) and 
                            lager['typ'] == LagerTyp.EINSPANNUNG):
                            balken.lager.remove(lager)
                            return
        elif existing_lager['typ'] == LagerTyp.GELENK:
            # Nach Gelenk: Lager entfernen (vollständiger Zyklus)
            for balken in self.balken_liste:
                for lager in balken.lager[:]:
                    if (self.points_close(lager['position'], position, tolerance=15) and 
                        lager['typ'] == LagerTyp.GELENK):
                        balken.lager.remove(lager)
                        return
    
    def get_balken_at_position(self, position):
        """Gibt alle Balken zurück, die an der Position enden"""
        balken_at_pos = []
        for balken in self.balken_liste:
            if (self.points_close(balken.start, position, tolerance=15) or 
                self.points_close(balken.end, position, tolerance=15)):
                balken_at_pos.append(balken)
        return balken_at_pos
    
    def validate_and_fix_lager(self):
        """Validiert und korrigiert alle Lager entsprechend den Regeln"""
        # Sammle alle Lagerpositionen
        lager_positions = {}
        
        # Sammle alle Lager gruppiert nach Position
        for balken in self.balken_liste:
            for lager in balken.lager[:]:  # Kopie der Liste für sichere Iteration
                pos_key = (round(lager['position'][0]), round(lager['position'][1]))
                
                if pos_key not in lager_positions:
                    lager_positions[pos_key] = []
                lager_positions[pos_key].append((balken, lager))
        
        # Verarbeite jede Position
        for pos_key, lager_list in lager_positions.items():
            position = (pos_key[0], pos_key[1])
            connected_balken = self.get_balken_at_position(position)
            
            # Regel 1: Lager nur an Balkenenden
            valid_lager = []
            for balken, lager in lager_list:
                if (self.points_close(lager['position'], balken.start, tolerance=15) or 
                    self.points_close(lager['position'], balken.end, tolerance=15)):
                    valid_lager.append((balken, lager))
                else:
                    # Lager nicht am Ende - entfernen
                    balken.lager.remove(lager)
            
            if not valid_lager:
                continue
            
            # Regel 2: Nur ein Lager pro Position
            if len(valid_lager) > 1:
                # Behalte nur das erste Lager, entferne die anderen
                for i, (balken, lager) in enumerate(valid_lager):
                    if i > 0:
                        balken.lager.remove(lager)
                valid_lager = [valid_lager[0]]
            
            # Regel 3: Gelenke nur an Verbindungen, nicht an freien Enden
            if valid_lager:
                balken, lager = valid_lager[0]
                is_free_end = len(connected_balken) == 1
                
                if lager['typ'] == LagerTyp.GELENK and is_free_end:
                    # Gelenk an freiem Ende - zu Loslager ändern
                    lager['typ'] = LagerTyp.LOSLAGER
    
    def handle_lasten_mode(self, pos):
        # Balken auswählen und Punktlast hinzufügen
        for balken in self.balken_liste:
            if balken.point_on_beam(pos):
                self.input_active = True
                self.input_position = self.project_point_on_beam(pos, balken)
                self.selected_balken = balken
                break
    
    def handle_punktlast_mode_click(self, pos, button=1):
        """Start des Punktlast-Ziehens - nur wenn auf einem Balken geklickt wird"""
        pos = self.snap_to_grid(pos)
        
        # Prüfen ob der Klick auf einem Balken erfolgt - erhöhte Toleranz
        clicked_balken = None
        for balken in self.balken_liste:
            if balken.point_on_beam(pos, tolerance=25):
                clicked_balken = balken
                break
        
        if clicked_balken:
            # Punktlast nur starten wenn auf Balken geklickt
            self.dragging_punktlast = True
            self.punktlast_start = self.project_point_on_beam(pos, clicked_balken)
            self.punktlast_balken = clicked_balken
        else:
            # Kein Balken getroffen - Punktlast nicht starten
            self.dragging_punktlast = False
            self.punktlast_start = None
            self.punktlast_balken = None
    
    def handle_punktlast_mode_release(self, pos):
        """Ende des Punktlast-Ziehens - nur wenn vorher auf Balken geklickt wurde"""
        if self.dragging_punktlast and self.punktlast_start and self.punktlast_balken:
            pos = self.snap_to_grid(pos)
            
            # Berechne Kraftvektor (beliebige Richtung möglich)
            dx = pos[0] - self.punktlast_start[0]
            dy = pos[1] - self.punktlast_start[1]
            
            # Mindest-Kraft prüfen
            kraft_betrag = math.sqrt(dx**2 + dy**2)
            if kraft_betrag > 5:  # Mindest-Kraft 5px = 5N
                
                # Prüfe ob bereits eine Punktlast am Balken an dieser Position existiert
                existing_last = None
                for last in self.punktlast_balken.punktlasten:
                    if (abs(last['position'][0] - self.punktlast_start[0]) < 10 and 
                        abs(last['position'][1] - self.punktlast_start[1]) < 10):
                        existing_last = last
                        break
                
                if existing_last:
                    # Kraft zu existierender Last hinzufügen/ersetzen
                    existing_last['kraft_x'] = dx
                    existing_last['kraft_y'] = dy
                    existing_last['kraft_betrag'] = kraft_betrag
                else:
                    # Neue Punktlast am Balken erstellen
                    self.punktlast_balken.punktlasten.append({
                        'position': self.punktlast_start,
                        'kraft_x': dx,
                        'kraft_y': dy,
                        'kraft_betrag': kraft_betrag
                    })
        
        self.dragging_punktlast = False
        self.punktlast_start = None
        self.punktlast_balken = None
    
    def handle_linienlast_mode(self, pos):
        """3-Klick-System für Linienlasten - Start und Ende müssen auf demselben Balken sein"""
        pos = self.snap_to_grid(pos)
        
        if self.linienlast_state == 0:  # Start setzen
            for balken in self.balken_liste:
                if balken.point_on_beam(pos, tolerance=25):
                    self.linienlast_start = self.project_point_on_beam(pos, balken)
                    self.linienlast_balken = balken
                    self.linienlast_state = 1
                    break
        
        elif self.linienlast_state == 1:  # Ende setzen (muss auf demselben Balken sein)
            if (self.linienlast_balken and 
                self.linienlast_balken.point_on_beam(pos, tolerance=25)):
                end_pos = self.project_point_on_beam(pos, self.linienlast_balken)
                # Prüfen dass Start und Ende unterschiedlich sind
                dx = end_pos[0] - self.linienlast_start[0]
                dy = end_pos[1] - self.linienlast_start[1]
                if math.sqrt(dx**2 + dy**2) > 10:  # Mindestlänge
                    self.linienlast_end = end_pos
                    self.linienlast_state = 2
            else:
                # Nicht auf demselben Balken - zurück zu Start
                self.reset_linienlast_state()
        
        elif self.linienlast_state == 2:  # Kraft setzen (senkrecht zum Balken)
            # Berechne senkrechte Distanz zum Balken
            balken = self.linienlast_balken
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge > 0:
                # Normierter Normalvektor (senkrecht zum Balken)
                normal_x = -dy / balken_laenge
                normal_y = dx / balken_laenge
                
                # Vektor vom Balken zum Klickpunkt
                start_to_click_x = pos[0] - self.linienlast_start[0]
                start_to_click_y = pos[1] - self.linienlast_start[1]
                
                # Projektion auf die Normale = Kraftbetrag
                kraft = start_to_click_x * normal_x + start_to_click_y * normal_y
                
                if abs(kraft) > 5:  # Mindest-Kraft 5N
                    self.linienlast_balken.linienlasten.append({
                        'start': self.linienlast_start,
                        'end': self.linienlast_end,
                        'kraft': kraft
                    })
            
            # Reset
            self.reset_linienlast_state()
    
    def handle_bearbeiten_mode(self, pos):
        """Handler für Bearbeiten-Modus"""
        # Zuerst alle Auswahlen zurücksetzen
        for balken in self.balken_liste:
            balken.selected = False
            balken.selected_lager = None
            balken.selected_punktlast = None
            balken.selected_linienlast = None
        self.selected_balken = None
        
        # Suche nach angeklicktem Element
        for balken in self.balken_liste:
            # Prüfe Lager
            clicked_lager = balken.get_clicked_lager(pos)
            if clicked_lager:
                balken.selected_lager = clicked_lager
                self.selected_balken = balken
                return
            
            # Prüfe Punktlasten
            clicked_punktlast = balken.get_clicked_punktlast(pos)
            if clicked_punktlast:
                balken.selected_punktlast = clicked_punktlast
                self.selected_balken = balken
                return
            
            # Prüfe Linienlasten
            clicked_linienlast = balken.get_clicked_linienlast(pos)
            if clicked_linienlast:
                balken.selected_linienlast = clicked_linienlast
                self.selected_balken = balken
                return
            
            # Prüfe Balken selbst
            if balken.point_on_beam(pos):
                balken.selected = True
                self.selected_balken = balken
                return
    
    def delete_selected_element(self):
        """Löscht das ausgewählte Element"""
        if not self.selected_balken:
            return
        
        balken = self.selected_balken
        
        # Löschen je nach Auswahl
        if balken.selected_lager:
            balken.lager.remove(balken.selected_lager)
            balken.selected_lager = None
        elif balken.selected_punktlast:
            balken.punktlasten.remove(balken.selected_punktlast)
            balken.selected_punktlast = None
        elif balken.selected_linienlast:
            balken.linienlasten.remove(balken.selected_linienlast)
            balken.selected_linienlast = None
        elif balken.selected:
            # Ganzen Balken löschen
            self.balken_liste.remove(balken)
            self.selected_balken = None
            return
        
        # Auswahl zurücksetzen
        self.selected_balken = None
    
    def get_lager_wertigkeit(self, lager_typ):
        """Gibt die Lagerwertigkeit zurück"""
        if lager_typ == LagerTyp.LOSLAGER:
            return 1  # nur F_z
        elif lager_typ == LagerTyp.FESTLAGER:
            return 2  # F_x, F_z
        elif lager_typ == LagerTyp.PARALLELFUEHRUNG:
            return 2  # F_x, M
        elif lager_typ == LagerTyp.SCHIEBEHUELSE:
            return 2  # F_z, M
        elif lager_typ == LagerTyp.EINSPANNUNG:
            return 3  # F_x, F_z, M
        elif lager_typ == LagerTyp.GELENK:
            return 2  # F_x, F_z (aber M = 0)
        return 0
    
    def ist_system_statisch_bestimmt(self):
        """Prüft statische Bestimmtheit: 3n = s + v"""
        if not self.balken_liste:
            return False, "Keine Balken vorhanden"
        
        n = len(self.balken_liste)  # Anzahl starrer Körper (Balken)
        
        # Lagerwertigkeiten s berechnen
        s = 0
        for balken in self.balken_liste:
            for lager in balken.lager:
                s += self.get_lager_wertigkeit(lager['typ'])
        
        # Gelenkwertigkeiten v berechnen (vereinfacht: Anzahl Gelenke * 2)
        v = 0
        gelenk_positionen = set()
        for balken in self.balken_liste:
            for lager in balken.lager:
                if lager['typ'] == LagerTyp.GELENK:
                    pos_key = (round(lager['position'][0]), round(lager['position'][1]))
                    gelenk_positionen.add(pos_key)
        v = len(gelenk_positionen) * 2
        
        freiheitsgrade_links = 3 * n
        freiheitsgrade_rechts = s + v
        
        if freiheitsgrade_links == freiheitsgrade_rechts:
            return True, f"System statisch bestimmt (3×{n} = {s}+{v})"
        elif freiheitsgrade_links > freiheitsgrade_rechts:
            return False, f"System kinematisch (3×{n} > {s}+{v})"
        else:
            return False, f"System statisch unbestimmt (3×{n} < {s}+{v})"
    
    def count_balken_connections(self):
        """Zählt Verbindungen zwischen Balken (vereinfacht)"""
        verbindungen = 0
        for i, balken1 in enumerate(self.balken_liste):
            for j, balken2 in enumerate(self.balken_liste[i+1:], i+1):
                # Prüfe ob Balken sich an Endpunkten treffen
                if (self.points_close(balken1.start, balken2.start, 10) or
                    self.points_close(balken1.start, balken2.end, 10) or
                    self.points_close(balken1.end, balken2.start, 10) or
                    self.points_close(balken1.end, balken2.end, 10)):
                    verbindungen += 1
        return verbindungen
    
    def points_close(self, point1, point2, tolerance):
        """Prüft ob zwei Punkte nahe beieinander liegen"""
        return abs(point1[0] - point2[0]) <= tolerance and abs(point1[1] - point2[1]) <= tolerance
    
    def project_point_on_beam(self, point, balken):
        """Projiziert einen Punkt auf den Balken"""
        x1, y1 = balken.start
        x2, y2 = balken.end
        px, py = point
        
        # Projektion auf die Linie
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / 
                       ((x2 - x1)**2 + (y2 - y1)**2)))
        
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return (int(proj_x), int(proj_y))
    
    def point_on_line(self, start, end, point, tolerance=10):
        """Prüft ob ein Punkt nahe einer Linie liegt"""
        x1, y1 = start
        x2, y2 = end
        px, py = point
        
        # Distanz Punkt zu Linie berechnen
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return abs(px - x1) <= tolerance and abs(py - y1) <= tolerance
        
        distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_length
        
        # Prüfen ob Punkt zwischen Start und End liegt
        dot_product = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        length_squared = (x2 - x1)**2 + (y2 - y1)**2
        
        if 0 <= dot_product <= length_squared and distance <= tolerance:
            return True
        return False
    
    def berechne_schnittkraefte(self):
        """Automatische Schnittkraftberechnung für statisch bestimmte Systeme"""
        # Nur wenn das System statisch bestimmt ist
        bestimmt, _ = self.ist_system_statisch_bestimmt()
        if not bestimmt:
            return
            
        # Für jeden Balken einzeln prüfen und Kraftverläufe zeichnen
        for balken in self.balken_liste:
            # Prüfe ob dieser spezifische Balken Lasten hat
            if self.ist_balken_belastet(balken):
                self.draw_lokales_koordinatensystem(balken)
                self.draw_schnittkraftverlaeufe(balken)
    
    def ist_balken_belastet(self, balken):
        """Prüft ob ein Balken Lasten oder Lager hat"""
        return (len(balken.lager) > 0 or 
                len(balken.punktlasten) > 0 or 
                len(balken.linienlasten) > 0)
    
    def handle_slider_click(self, pos):
        """Behandelt Klicks auf den Skalierungs-Slider"""
        if self.scale_slider['handle_rect'].collidepoint(pos):
            self.scale_slider['dragging'] = True
            return True
        return False
    
    def handle_slider_drag(self, pos):
        """Behandelt das Ziehen des Slider-Handles"""
        if not self.scale_slider['dragging']:
            return
        
        # Neue Handle-Position berechnen (innerhalb der Slider-Grenzen)
        slider_rect = self.scale_slider['rect']
        handle_rect = self.scale_slider['handle_rect']
        
        # X-Position des Handles innerhalb des Sliders
        min_x = slider_rect.x
        max_x = slider_rect.x + slider_rect.width - handle_rect.width
        new_x = max(min_x, min(max_x, pos[0] - handle_rect.width // 2))
        
        # Handle-Position aktualisieren
        handle_rect.x = new_x
        
        # Skalierungswert berechnen
        progress = (new_x - min_x) / (max_x - min_x)
        min_scale = self.scale_slider['min_scale']
        max_scale = self.scale_slider['max_scale']
        self.scale_slider['current_scale'] = min_scale + progress * (max_scale - min_scale)
    
    def draw_lokales_koordinatensystem(self, balken):
        """Zeichnet das lokale Koordinatensystem am Balkenanfang"""
        start_x, start_y = balken.start
        end_x, end_y = balken.end
        
        # Balken-Geometrie berechnen
        dx = end_x - start_x
        dy = end_y - start_y
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return
        
        # Einheitsvektoren des lokalen Koordinatensystems
        # x-Achse: entlang Balken (Laufrichtung)
        ex_x = dx / balken_laenge
        ex_y = dy / balken_laenge
        
        # z-Achse: senkrecht nach unten (Querkraftrichtung)
        ez_x = ex_y   # Senkrecht zur x-Achse (nach unten positiv)
        ez_y = -ex_x
        
        # Koordinatensystem-Pfeile zeichnen
        pfeil_laenge = 30
        
        # x-Achse (rot) - Laufrichtung
        x_end = (start_x + pfeil_laenge * ex_x, start_y + pfeil_laenge * ex_y)
        pygame.draw.line(self.screen, (255, 0, 0), balken.start, x_end, 2)
        self.draw_pfeilspitze(self.screen, balken.start, x_end, (255, 0, 0))
        
        # z-Achse (blau) - Querkraftrichtung
        z_end = (start_x + pfeil_laenge * ez_x, start_y + pfeil_laenge * ez_y)
        pygame.draw.line(self.screen, (0, 0, 255), balken.start, z_end, 2)
        self.draw_pfeilspitze(self.screen, balken.start, z_end, (0, 0, 255))
        
        # Beschriftung
        font = pygame.font.Font(None, 20)
        x_text = font.render("x", True, (255, 0, 0))
        z_text = font.render("z", True, (0, 0, 255))
        
        self.screen.blit(x_text, (x_end[0] + 5, x_end[1] - 10))
        self.screen.blit(z_text, (z_end[0] + 5, z_end[1] - 10))
    
    def draw_pfeilspitze(self, screen, start, end, color):
        """Zeichnet eine Pfeilspitze"""
        # Vektor vom Start zum Ende
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        laenge = math.sqrt(dx*dx + dy*dy)
        
        if laenge > 0:
            # Einheitsvektor
            ux = dx / laenge
            uy = dy / laenge
            
            # Pfeilspitzen-Punkte
            spitze_laenge = 8
            spitze_winkel = 0.5
            
            p1_x = end[0] - spitze_laenge * (ux * math.cos(spitze_winkel) - uy * math.sin(spitze_winkel))
            p1_y = end[1] - spitze_laenge * (ux * math.sin(spitze_winkel) + uy * math.cos(spitze_winkel))
            
            p2_x = end[0] - spitze_laenge * (ux * math.cos(-spitze_winkel) - uy * math.sin(-spitze_winkel))
            p2_y = end[1] - spitze_laenge * (ux * math.sin(-spitze_winkel) + uy * math.cos(-spitze_winkel))
            
            pygame.draw.polygon(screen, color, [end, (p1_x, p1_y), (p2_x, p2_y)])
    def draw_schnittkraftverlaeufe(self, balken):
        """Zeichnet die Schnittkraftverläufe N, Q, M direkt über dem Balken"""
        start_x, start_y = balken.start
        end_x, end_y = balken.end
        
        # Balken-Geometrie berechnen
        dx = end_x - start_x
        dy = end_y - start_y
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge < 50:  # Balken zu kurz
            return
        
        # Einheitsvektoren des lokalen Koordinatensystems
        # x-Achse: entlang Balken (Laufrichtung)
        ex_x = dx / balken_laenge
        ex_y = dy / balken_laenge
        
        # z-Achse: senkrecht nach oben (für positive Darstellung über dem Balken)
        ez_x = -ex_y   # Senkrecht zur Balkenrichtung (nach oben positiv für Darstellung)
        ez_y = ex_x
        
        # Skalierungsfaktor für Darstellung (vom Slider)
        skalierung = self.scale_slider['current_scale']
        
        # Anzahl Punkte entlang des Balkens
        anzahl_punkte = max(20, int(balken_laenge / 5))
        
        # Alle Schnittkräfte verwenden die Balkenlinie als Nulllinie
        # Nur die Farbe unterscheidet N (rot), Q (grün), M (magenta)
        
        # 1. Normalkraft N (rot) - Balkenlinie als Nulllinie
        n_punkte = []
        for i in range(anzahl_punkte + 1):
            t = i / anzahl_punkte
            # Position auf dem Balken (Nulllinie)
            pos_x = start_x + t * dx
            pos_y = start_y + t * dy
            
            n_wert = self.get_normalkraft_N_korrekt(balken, t)
            
            # Punkt senkrecht über/unter dem Balken
            offset_x = pos_x + n_wert * skalierung * ez_x
            offset_y = pos_y + n_wert * skalierung * ez_y
            n_punkte.append((offset_x, offset_y))
        
        if len(n_punkte) > 1:
            # N-Verlauf zeichnen
            pygame.draw.lines(self.screen, (255, 0, 0), False, n_punkte, 2)
        
        # 2. Querkraft Q (grün) - Balkenlinie als Nulllinie
        q_punkte = []
        for i in range(anzahl_punkte + 1):
            t = i / anzahl_punkte
            # Position auf dem Balken (Nulllinie)
            pos_x = start_x + t * dx
            pos_y = start_y + t * dy
            
            q_wert = self.get_querkraft_Q_korrekt(balken, t)
            
            # Punkt senkrecht über/unter dem Balken
            offset_x = pos_x + q_wert * skalierung * ez_x
            offset_y = pos_y + q_wert * skalierung * ez_y
            q_punkte.append((offset_x, offset_y))
        
        if len(q_punkte) > 1:
            # Q-Verlauf zeichnen
            pygame.draw.lines(self.screen, (0, 255, 0), False, q_punkte, 2)
        
        # 3. Biegemoment M (magenta) - Balkenlinie als Nulllinie
        m_punkte = []
        for i in range(anzahl_punkte + 1):
            t = i / anzahl_punkte
            # Position auf dem Balken (Nulllinie)
            pos_x = start_x + t * dx
            pos_y = start_y + t * dy
            
            m_wert = self.get_biegemoment_M_korrekt(balken, t)
            
            # Punkt senkrecht über/unter dem Balken
            offset_x = pos_x + m_wert * skalierung * 0.8 * ez_x  # Etwas kleinere Skalierung für M
            offset_y = pos_y + m_wert * skalierung * 0.8 * ez_y
            m_punkte.append((offset_x, offset_y))
        
        if len(m_punkte) > 1:
            # M-Verlauf zeichnen
            pygame.draw.lines(self.screen, (255, 0, 255), False, m_punkte, 2)
        
        # Labels neben dem Balkenanfang - alle bei gleicher Höhe
        font = pygame.font.Font(None, 20)
        label_offset = 60
        
        n_text = font.render("N", True, (255, 0, 0))
        q_text = font.render("Q", True, (0, 255, 0))
        m_text = font.render("M", True, (255, 0, 255))
        
        # Labels nebeneinander positionieren (alle auf gleicher Höhe)
        label_x = start_x - label_offset * ex_x + 10 * ez_x
        label_y = start_y - label_offset * ex_y + 10 * ez_y
        
        self.screen.blit(n_text, (label_x, label_y))
        self.screen.blit(q_text, (label_x + 20, label_y))  # Q etwas versetzt
        self.screen.blit(m_text, (label_x + 40, label_y))  # M weiter versetzt
    
    def get_normalkraft_N_korrekt(self, balken, t):
        """Korrekte Berechnung der Normalkraft N mit Ursprung am ersten Klick"""
        # t = 0 entspricht dem ersten Klick (balken.start)
        # t = 1 entspricht dem zweiten Klick (balken.end)
        
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        # Einheitsvektor in Balkenrichtung (x-Achse des lokalen Systems)
        ex_x = dx / balken_laenge
        ex_y = dy / balken_laenge
        
        N = 0
        
        # Alle Kräfte RECHTS vom Schnitt (t bis 1.0) berücksichtigen
        # 1. Punktlasten rechts vom Schnitt
        for last in balken.punktlasten:
            last_t = self.get_parameter_on_balken(last['position'], balken)
            if last_t >= t:  # Rechts vom Schnitt
                kraft_x = last.get('kraft_x', 0)
                kraft_y = last.get('kraft_y', 0)
                
                # Fallback für alte Punktlasten
                if kraft_x == 0 and kraft_y == 0:
                    kraft_y = last.get('kraft', 0)
                
                # Projektion auf Balkenachse (positive Richtung = Zug)
                N += kraft_x * ex_x + kraft_y * ex_y
        
        # 2. Linienlasten rechts vom Schnitt
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            
            # Nur der Teil rechts vom Schnitt
            schnitt_start = max(start_t, t)
            schnitt_end = end_t
            
            if schnitt_start < schnitt_end:
                laenge_rechts = (schnitt_end - schnitt_start) * balken_laenge
                resultierende = last['kraft'] * laenge_rechts / 100
                
                # Linienlast wirkt normalerweise senkrecht, aber kann axiale Komponente haben
                # Vereinfacht: nehme nur senkrechte Linienlasten an
                # N += 0  # Linienlasten erzeugen normalerweise keine Normalkraft
        
        # 3. Lagerreaktionen rechts vom Schnitt
        for lager in balken.lager:
            lager_t = self.get_parameter_on_balken(lager['position'], balken)
            if lager_t >= t:  # Rechts vom Schnitt
                lager_N = self.get_lagerreaktion_N_korrekt(balken, lager)
                N += lager_N
        
        return N
    
    def get_querkraft_Q_korrekt(self, balken, t):
        """Korrekte Berechnung der Querkraft Q mit Ursprung am ersten Klick"""
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        # Einheitsvektor senkrecht zur Balkenachse (z-Achse des lokalen Systems)
        ez_x = -dy / balken_laenge   # Senkrecht zur Balkenrichtung (nach oben positiv)
        ez_y = dx / balken_laenge
        
        Q = 0
        
        # Alle Kräfte RECHTS vom Schnitt berücksichtigen
        # 1. Punktlasten rechts vom Schnitt
        for last in balken.punktlasten:
            last_t = self.get_parameter_on_balken(last['position'], balken)
            if last_t >= t:  # Rechts vom Schnitt
                kraft_x = last.get('kraft_x', 0)
                kraft_y = last.get('kraft_y', 0)
                
                # Fallback für alte Punktlasten
                if kraft_x == 0 and kraft_y == 0:
                    kraft_y = last.get('kraft', 0)
                
                # Projektion auf Querkraftrichtung (Vorzeichen umkehren für Schnittkraft)
                Q -= kraft_x * ez_x + kraft_y * ez_y
        
        # 2. Linienlasten rechts vom Schnitt
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            
            schnitt_start = max(start_t, t)
            schnitt_end = end_t
            
            if schnitt_start < schnitt_end:
                laenge_rechts = (schnitt_end - schnitt_start) * balken_laenge
                resultierende = last['kraft'] * laenge_rechts / 100
                
                # Linienlast wirkt senkrecht zum Balken (Vorzeichen umkehren für Schnittkraft)
                Q -= resultierende
        
        # 3. Lagerreaktionen rechts vom Schnitt
        for lager in balken.lager:
            lager_t = self.get_parameter_on_balken(lager['position'], balken)
            if lager_t >= t:  # Rechts vom Schnitt
                lager_Q = self.get_lagerreaktion_Q_korrekt(balken, lager)
                Q -= lager_Q
        
        return Q
    
    def get_biegemoment_M_korrekt(self, balken, t):
        """Biegemoment nach korrekter Statik: M(x) durch Integration der Querkraft"""
        dx = balken.end[0] - balken.start[0] 
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        # Einheitsvektor senkrecht zur Balkenachse (für Querkraft)
        ez_x = -dy / balken_laenge   # Senkrecht zur Balkenrichtung (nach oben positiv)
        ez_y = dx / balken_laenge
        
        # Momentenverlauf durch Integration der Querkraft von x=0 bis x=t
        M = 0
        
        # Startwert: Einspannmoment am Anfang (falls vorhanden)
        start_lager = [l for l in balken.lager 
                      if self.get_parameter_on_balken(l['position'], balken) <= 0.01]
        
        if start_lager and start_lager[0]['typ'] == LagerTyp.EINSPANNUNG:
            # Bei Einspannung am Anfang: Einspannmoment als Startwert
            M = self.get_einspannmoment_kragarm(balken, start_lager[0])
        
        # Integration: M(x) = M₀ + ∫Q(ξ)dξ von 0 bis x
        # Vereinfacht durch diskrete Summation
        schritte = 100
        dx_schritt = t / schritte if schritte > 0 else 0
        
        for i in range(schritte):
            xi = i * dx_schritt
            Q_xi = self.get_querkraft_Q_korrekt(balken, xi)
            M += Q_xi * dx_schritt * balken_laenge  # Integration: Summe Q*dx
        
        # Randbedingungen prüfen
        # Am freien Ende: M = 0
        if t >= 0.99:  # Ende des Balkens
            end_lager = [l for l in balken.lager 
                        if self.get_parameter_on_balken(l['position'], balken) >= 0.99]
            
            if not end_lager:
                # Freies Ende
                M = 0
            elif end_lager[0]['typ'] in [LagerTyp.GELENK, LagerTyp.LOSLAGER, LagerTyp.FESTLAGER]:
                # Diese Lager können keine Momente übertragen
                M = 0
        
        # Am Gelenk/Loslager/Festlager: M = 0
        for lager in balken.lager:
            lager_t = self.get_parameter_on_balken(lager['position'], balken)
            if abs(lager_t - t) < 0.01:  # An der Lagerposition
                if lager['typ'] in [LagerTyp.GELENK, LagerTyp.LOSLAGER, LagerTyp.FESTLAGER]:
                    M = 0
        
        return M / 50  # Skalierung für Darstellung
    
    def get_einspannmoment_kragarm(self, balken, einspannung):
        """Berechnet das Einspannmoment für einen Kragarm"""
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        # Einheitsvektor senkrecht zur Balkenachse
        ez_x = -dy / balken_laenge   # Senkrecht zur Balkenrichtung (nach oben positiv)
        ez_y = dx / balken_laenge
        
        einspann_t = self.get_parameter_on_balken(einspannung['position'], balken)
        einspann_pos = einspann_t * balken_laenge
        
        M_einspann = 0
        
        # Momente aller Kräfte um die Einspannstelle
        # 1. Punktlasten
        for last in balken.punktlasten:
            last_t = self.get_parameter_on_balken(last['position'], balken)
            last_pos = last_t * balken_laenge
            hebelarm = last_pos - einspann_pos
            
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            # Querkraft-Komponente
            Q_kraft = kraft_x * ez_x + kraft_y * ez_y
            M_einspann += Q_kraft * hebelarm
        
        # 2. Linienlasten
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            
            # Schwerpunkt der Linienlast
            schwerpunkt_t = (start_t + end_t) / 2
            schwerpunkt_pos = schwerpunkt_t * balken_laenge
            hebelarm = schwerpunkt_pos - einspann_pos
            
            # Resultierende der Linienlast
            laenge = (end_t - start_t) * balken_laenge
            resultierende = last['kraft'] * laenge / 100
            
            M_einspann += resultierende * hebelarm
        
        # Das Einspannmoment muss die äußeren Momente ausgleichen
        return -M_einspann
    
    def get_lagerreaktion_N_korrekt(self, balken, lager):
        """Berechnet die Normalkraft-Komponente einer Lagerreaktion"""
        if lager['typ'] == LagerTyp.GELENK:
            return 0  # Gelenke übertragen keine Kräfte
        
        # Für eine korrekte Berechnung müsste das gesamte System gelöst werden
        # Vereinfacht: nur bei axialen Lasten relevant
        return 0
    
    def get_lagerreaktion_Q_korrekt(self, balken, lager):
        """Berechnet die Querkraft-Komponente einer Lagerreaktion"""
        if lager['typ'] == LagerTyp.GELENK:
            return 0  # Gelenke übertragen keine Kräfte
        
        # Für vereinfachte Berechnung: Annahme dass Lager die lokalen Lasten aufnimmt
        # In einer vollständigen Implementierung würde hier das Gleichungssystem gelöst
        
        # Schätzung basierend auf lokalen Lasten
        summe_lasten = 0
        
        # Punktlasten auf dem Balken
        for last in balken.punktlasten:
            kraft_y = last.get('kraft_y', 0)
            if kraft_y == 0:  # Fallback
                kraft_y = last.get('kraft', 0)
            summe_lasten += kraft_y
        
        # Linienlasten auf dem Balken
        for last in balken.linienlasten:
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            laenge = (end_t - start_t) * balken_laenge
            resultierende = last['kraft'] * laenge / 100
            summe_lasten += resultierende
        
        # Vereinfachte Aufteilung auf Lager
        anzahl_tragende_lager = sum(1 for l in balken.lager 
                                   if l['typ'] not in [LagerTyp.GELENK])
        
        if anzahl_tragende_lager > 0:
            return -summe_lasten / anzahl_tragende_lager
        
        return 0
    
    def get_lagerreaktion_M_korrekt(self, balken, lager):
        """Berechnet das Einspannmoment einer Lagerreaktion korrekt nach Statik"""
        if lager['typ'] not in [LagerTyp.EINSPANNUNG, LagerTyp.PARALLELFUEHRUNG, LagerTyp.SCHIEBEHUELSE]:
            return 0  # Nur diese Lager können Momente übertragen
        
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        # Einheitsvektoren
        ez_x = -dy / balken_laenge   # z-Richtung (Querkraftrichtung, nach oben positiv)
        ez_y = dx / balken_laenge
        
        lager_t = self.get_parameter_on_balken(lager['position'], balken)
        lager_position = lager_t * balken_laenge  # Absolute Position des Lagers
        
        M_einspann = 0
        
        # Bei Kragarm (Einspannung am Anfang, freies Ende):
        # Das Einspannmoment ergibt sich aus dem Momentengleichgewicht
        
        # 1. Momente von Punktlasten um die Einspannstelle
        for last in balken.punktlasten:
            last_t = self.get_parameter_on_balken(last['position'], balken)
            last_position = last_t * balken_laenge
            
            # Hebelarm = Abstand von Lager zur Last
            hebelarm = last_position - lager_position
            
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            
            # Fallback für alte Punktlasten
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            # Nur Querkraft-Komponente erzeugt Biegemoment
            Q_kraft = kraft_x * ez_x + kraft_y * ez_y
            M_einspann += Q_kraft * hebelarm
        
        # 2. Momente von Linienlasten um die Einspannstelle
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            
            # Schwerpunkt der Linienlast
            schwerpunkt_t = (start_t + end_t) / 2
            schwerpunkt_position = schwerpunkt_t * balken_laenge
            
            # Hebelarm vom Lager zum Schwerpunkt
            hebelarm = schwerpunkt_position - lager_position
            
            # Resultierende Kraft der gesamten Linienlast
            laenge = (end_t - start_t) * balken_laenge
            resultierende = last['kraft'] * laenge / 100
            
            M_einspann += resultierende * hebelarm
        
        # Das Einspannmoment muss die äußeren Momente ausgleichen
        return -M_einspann
    
    def get_parameter_on_balken(self, position, balken):
        """Berechnet den Parameter t (0-1) für eine Position auf dem Balken"""
        if not position or not balken:
            return 0
        
        # Balkenvektor
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge_quadrat = dx**2 + dy**2
        
        if balken_laenge_quadrat == 0:
            return 0
        
        # Vektor vom Balkenanfang zur Position
        px = position[0] - balken.start[0]
        py = position[1] - balken.start[1]
        
        # Projektion: t = (P-Start) · (End-Start) / |End-Start|²
        t = (px * dx + py * dy) / balken_laenge_quadrat
        
        # Begrenze auf [0, 1]
        return max(0, min(1, t))
    def get_lagerreaktion_N(self, balken, lager):
        """Berechnet die Normalkraft-Komponente der Lagerreaktion"""
        if lager['typ'] == LagerTyp.LOSLAGER:
            return 0  # F_x = 0
        elif lager['typ'] == LagerTyp.SCHIEBEHUELSE:
            return 0  # F_x = 0
        elif lager['typ'] == LagerTyp.GELENK:
            return 0  # Keine Kraftübertragung
        else:
            # Festlager, Parallelführung, Einspannung haben N-Reaktion
            return self.berechne_lagerreaktion_statisch(balken, lager, 'N')
    
    def get_lagerreaktion_Q(self, balken, lager):
        """Berechnet die Querkraft-Komponente der Lagerreaktion"""
        if lager['typ'] == LagerTyp.PARALLELFUEHRUNG:
            return 0  # F_z = 0
        elif lager['typ'] == LagerTyp.GELENK:
            return 0  # Keine Kraftübertragung
        else:
            # Loslager, Festlager, Schiebehülse, Einspannung haben Q-Reaktion
            return self.berechne_lagerreaktion_statisch(balken, lager, 'Q')
    
    def get_lagerreaktion_M(self, balken, lager):
        """Berechnet die Moment-Komponente der Lagerreaktion"""
        if lager['typ'] in [LagerTyp.LOSLAGER, LagerTyp.FESTLAGER, LagerTyp.GELENK]:
            return 0  # M = 0
        else:
            # Parallelführung, Schiebehülse, Einspannung haben M-Reaktion
            return self.berechne_einspannmoment_statisch(balken, lager)
    
    def berechne_lagerreaktion_statisch(self, balken, lager, komponente):
        """Vereinfachte statische Berechnung der Lagerreaktionen"""
        # Sammle alle äußeren Kräfte
        gesamt_kraft_x = 0
        gesamt_kraft_y = 0
        
        # Punktlasten
        for last in balken.punktlasten:
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            gesamt_kraft_x += kraft_x
            gesamt_kraft_y += kraft_y
        
        # Linienlasten
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge > 0:
                laenge = (end_t - start_t) * balken_laenge
                resultierende = last['kraft'] * laenge / 100
                
                # Linienlast wirkt senkrecht zum Balken
                ez_x = -dy / balken_laenge
                ez_y = dx / balken_laenge
                
                gesamt_kraft_x += resultierende * ez_x
                gesamt_kraft_y += resultierende * ez_y
        
        # Vereinfachte Gleichgewichtsberechnung
        reaktions_lager = [l for l in balken.lager 
                          if l['typ'] != LagerTyp.GELENK]
        
        if len(reaktions_lager) == 0:
            return 0
        
        # Balken-Koordinatensystem
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        ex_x = dx / balken_laenge  # x-Richtung
        ex_y = dy / balken_laenge
        ez_x = -dy / balken_laenge   # z-Richtung (nach oben positiv)
        ez_y = dx / balken_laenge
        
        if komponente == 'N':
            # Normalkraft-Komponente
            gesamt_N = gesamt_kraft_x * ex_x + gesamt_kraft_y * ex_y
            return -gesamt_N / len(reaktions_lager)
        elif komponente == 'Q':
            # Querkraft-Komponente
            gesamt_Q = gesamt_kraft_x * ez_x + gesamt_kraft_y * ez_y
            return -gesamt_Q / len(reaktions_lager)
        
        return 0
    
    def berechne_einspannmoment_statisch(self, balken, lager):
        """Berechnet das Einspannmoment statisch"""
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        lager_t = self.get_parameter_on_balken(lager['position'], balken)
        
        # Momentengleichgewicht um die Lagerstelle
        gesamt_moment = 0
        
        # Momente aller Punktlasten
        for last in balken.punktlasten:
            last_t = self.get_parameter_on_balken(last['position'], balken)
            hebelarm = abs(last_t - lager_t) * balken_laenge
            
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            # Nur Querkraft erzeugt Moment
            ez_x = -dy / balken_laenge
            ez_y = dx / balken_laenge
            
            Q_kraft = kraft_x * ez_x + kraft_y * ez_y
            
            if last_t > lager_t:
                gesamt_moment += Q_kraft * hebelarm
            else:
                gesamt_moment -= Q_kraft * hebelarm
        
        return -gesamt_moment
    
    def draw_vertikales_diagramm(self, basis_x, start_y, end_y, breite, label, color, wert_funktion):
        """Zeichnet ein Diagramm für vertikale Balken"""
        # Basis-Linie (vertikal)
        pygame.draw.line(self.screen, (100, 100, 100), (basis_x, start_y), (basis_x, end_y), 1)
        
        # Diagramm-Punkte berechnen
        punkte = []
        werte = []
        balken_laenge = abs(end_y - start_y)
        
        for i in range(int(balken_laenge) + 1):
            y = start_y + i
            # Relative Position auf dem Balken (0 bis 1)
            relative_pos = i / balken_laenge if balken_laenge > 0 else 0
            
            # Wert an dieser Position
            wert = wert_funktion(relative_pos)
            werte.append(wert)
        
        # Maximum für Normierung finden
        max_wert = max(abs(w) for w in werte) if werte else 1
        skalierung = breite / max_wert if max_wert > 0 else 1
        
        # Punkte mit normierter Skalierung erstellen
        for i, wert in enumerate(werte):
            y = start_y + i
            # X-Koordinate (normiert auf maximale Diagramm-Breite)
            x = basis_x + wert * skalierung
            punkte.append((x, y))
        
        # Diagramm zeichnen
        if len(punkte) > 1:
            pygame.draw.lines(self.screen, color, False, punkte, 2)
            
            # Fläche zwischen Nulllinie und Diagramm füllen
            if len(punkte) > 2:
                fill_punkte = punkte + [(basis_x, end_y), (basis_x, start_y)]
                pygame.draw.polygon(self.screen, (*color, 50), fill_punkte)
        
        # Label
        font = pygame.font.Font(None, 20)
        text = font.render(label, True, color)
        self.screen.blit(text, (basis_x + 5, start_y - 20))
        
        # Maximalwert anzeigen
        if max_wert > 0:
            max_text = font.render(f"Max: {max_wert:.1f}", True, color)
            self.screen.blit(max_text, (basis_x + breite + 5, start_y))
    
    def get_normalkraft_funktion_lokal(self, balken):
        """Gibt eine Funktion für die Normalkraft im lokalen Koordinatensystem zurück"""
        def normalkraft(t):
            # t ist Parameter von 0 bis 1 entlang des Balkens
            # Balken-Geometrie
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge == 0:
                return 0
                
            # Normalisierte Richtungsvektoren
            cos_alpha = dx / balken_laenge
            sin_alpha = dy / balken_laenge
            
            # Aktuelle Position entlang des Balkens
            x_pos = balken.start[0] + t * dx
            y_pos = balken.start[1] + t * dy
            
            gesamt_normalkraft = 0
            
            # Lagerreaktionen berücksichtigen (rechts vom Schnitt)
            for lager in balken.lager:
                lager_pos = lager['position']
                # Parameter des Lagers entlang des Balkens berechnen
                lager_t = self.get_parameter_on_balken(lager_pos, balken)
                
                if lager_t >= t:  # Lager rechts vom Schnitt
                    if lager['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG, LagerTyp.GELENK]:
                        # Vereinfachte Lagerreaktion in Balkenrichtung
                        lager_reaktion_x, lager_reaktion_y = self.berechne_lagerreaktion_komponenten(balken, lager)
                        # Projektion auf Balkenachse
                        normalkraft_anteil = lager_reaktion_x * cos_alpha + lager_reaktion_y * sin_alpha
                        gesamt_normalkraft += normalkraft_anteil
            
            # Punktlasten am Balken (rechts vom Schnitt)
            for last in balken.punktlasten:
                last_pos = last['position']
                last_t = self.get_parameter_on_balken(last_pos, balken)
                
                if last_t >= t:  # Last rechts vom Schnitt
                    kraft_x = last.get('kraft_x', 0)
                    kraft_y = last.get('kraft_y', 0)
                    
                    # Fallback für alte Punktlasten
                    if kraft_x == 0 and kraft_y == 0:
                        kraft_y = last.get('kraft', 0)
                    
                    # Projektion auf Balkenachse
                    normalkraft_anteil = kraft_x * cos_alpha + kraft_y * sin_alpha
                    gesamt_normalkraft += normalkraft_anteil
            
            # Linienlasten berücksichtigen (vereinfacht)
            for last in balken.linienlasten:
                start_t = self.get_parameter_on_balken(last['start'], balken)
                end_t = self.get_parameter_on_balken(last['end'], balken)
                
                # Bereich der Linienlast, der rechts vom Schnitt liegt
                schnitt_start = max(start_t, t)
                schnitt_end = end_t
                
                if schnitt_start < schnitt_end:
                    laenge_rechts = (schnitt_end - schnitt_start) * balken_laenge
                    resultierende = last['kraft'] * laenge_rechts / 100
                    # Projektion auf Balkenachse
                    normalkraft_anteil = resultierende * sin_alpha
                    gesamt_normalkraft += normalkraft_anteil
            
            return gesamt_normalkraft
        
        return normalkraft
    
    def get_querkraft_funktion_lokal(self, balken):
        """Gibt eine Funktion für die Querkraft im lokalen Koordinatensystem zurück"""
        def querkraft(t):
            # t ist Parameter von 0 bis 1 entlang des Balkens
            # Balken-Geometrie
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge == 0:
                return 0
                
            # Normalisierte Richtungsvektoren
            cos_alpha = dx / balken_laenge
            sin_alpha = dy / balken_laenge
            # Normale (senkrecht zum Balken)
            normal_x = -sin_alpha
            normal_y = cos_alpha
            
            gesamt_querkraft = 0
            
            # Lagerreaktionen berücksichtigen (rechts vom Schnitt)
            for lager in balken.lager:
                lager_pos = lager['position']
                lager_t = self.get_parameter_on_balken(lager_pos, balken)
                
                if lager_t >= t:  # Lager rechts vom Schnitt
                    if lager['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG, LagerTyp.GELENK]:
                        lager_reaktion_x, lager_reaktion_y = self.berechne_lagerreaktion_komponenten(balken, lager)
                        # Projektion auf Normale (senkrecht zur Balkenachse)
                        querkraft_anteil = lager_reaktion_x * normal_x + lager_reaktion_y * normal_y
                        gesamt_querkraft += querkraft_anteil
            
            # Punktlasten am Balken (rechts vom Schnitt)
            for last in balken.punktlasten:
                last_pos = last['position']
                last_t = self.get_parameter_on_balken(last_pos, balken)
                
                if last_t >= t:  # Last rechts vom Schnitt
                    kraft_x = last.get('kraft_x', 0)
                    kraft_y = last.get('kraft_y', 0)
                    
                    # Fallback für alte Punktlasten
                    if kraft_x == 0 and kraft_y == 0:
                        kraft_y = last.get('kraft', 0)
                    
                    # Projektion auf Normale
                    querkraft_anteil = kraft_x * normal_x + kraft_y * normal_y
                    gesamt_querkraft += querkraft_anteil
            
            # Linienlasten berücksichtigen
            for last in balken.linienlasten:
                start_t = self.get_parameter_on_balken(last['start'], balken)
                end_t = self.get_parameter_on_balken(last['end'], balken)
                
                schnitt_start = max(start_t, t)
                schnitt_end = end_t
                
                if schnitt_start < schnitt_end:
                    laenge_rechts = (schnitt_end - schnitt_start) * balken_laenge
                    resultierende = last['kraft'] * laenge_rechts / 100
                    # Projektion auf Normale
                    querkraft_anteil = resultierende * normal_y
                    gesamt_querkraft += querkraft_anteil
            
            return gesamt_querkraft
        
        return querkraft
    
    def get_moment_funktion_lokal(self, balken):
        """Gibt eine Funktion für das Biegemoment im lokalen Koordinatensystem zurück - korrekte Berechnung"""
        def moment(t):
            # t ist Parameter von 0 bis 1 entlang des Balkens
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge == 0:
                return 0
            
            # Aktuelle Position entlang des Balkens
            x_schnitt = balken_laenge * t
            
            # Moment durch Schnittprinzip berechnen:
            # M(x) = Summe aller Momente aller Kräfte links vom Schnitt um den Schnittpunkt
            gesamt_moment = 0
            
            # 1. Beiträge von Punktlasten (links vom Schnitt)
            for last in balken.punktlasten:
                last_t = self.get_parameter_on_balken(last['position'], balken)
                
                if last_t <= t:  # Last liegt links vom Schnitt
                    # Hebelarm = Abstand der Last zum Schnittpunkt
                    hebelarm = (t - last_t) * balken_laenge
                    
                    kraft_x = last.get('kraft_x', 0)
                    kraft_y = last.get('kraft_y', 0)
                    
                    # Fallback für alte Punktlasten
                    if kraft_x == 0 and kraft_y == 0:
                        kraft_y = last.get('kraft', 0)
                    
                    # Nur die Querkraft-Komponente (senkrecht zum Balken) erzeugt Biegemoment
                    # Balken-Koordinatensystem: x entlang Balken, y senkrecht dazu
                    normal_x = -dy / balken_laenge  # Richtung senkrecht zum Balken
                    normal_y = dx / balken_laenge
                    
                    # Querkraft-Komponente der Last
                    querkraft_komponente = kraft_x * normal_x + kraft_y * normal_y
                    
                    # Moment = Querkraft * Hebelarm (Vorzeichenkonvention beachten)
                    gesamt_moment += querkraft_komponente * hebelarm
            
            # 2. Beiträge von Linienlasten (links vom Schnitt)
            for last in balken.linienlasten:
                start_t = self.get_parameter_on_balken(last['start'], balken)
                end_t = self.get_parameter_on_balken(last['end'], balken)
                
                # Nur der Teil der Linienlast, der links vom Schnitt liegt
                schnitt_start = start_t
                schnitt_end = min(end_t, t)
                
                if schnitt_start < schnitt_end:
                    # Schwerpunkt der linksliegenden Linienlast
                    schwerpunkt_t = (schnitt_start + schnitt_end) / 2
                    hebelarm = (t - schwerpunkt_t) * balken_laenge
                    
                    # Resultierende Kraft der linksliegenden Linienlast
                    laenge_links = (schnitt_end - schnitt_start) * balken_laenge
                    resultierende_kraft = last['kraft'] * laenge_links / 100  # Skalierung
                    
                    # Linienlast wirkt senkrecht zum Balken
                    normal_y = dx / balken_laenge
                    querkraft_komponente = resultierende_kraft * normal_y
                    
                    # Moment = Resultierende * Hebelarm
                    gesamt_moment += querkraft_komponente * hebelarm
            
            # 3. Lagerreaktionen (links vom Schnitt)
            for lager in balken.lager:
                lager_t = self.get_parameter_on_balken(lager['position'], balken)
                
                if lager_t <= t:  # Lager liegt links vom Schnitt
                    # Momentenlagerreaktionen (nur bei Einspannung!)
                    if lager['typ'] == LagerTyp.EINSPANNUNG:
                        # Einspannmoment direkt hinzufügen
                        einspann_moment = self.berechne_einspannmoment_korrekt(balken, lager)
                        gesamt_moment += einspann_moment
                    
                    # Lagerreaktionskräfte als Einzelkräfte behandeln
                    # Alle Lagertypen außer Gelenk haben Kraftreaktionen
                    if lager['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG]:
                        hebelarm = (t - lager_t) * balken_laenge
                        
                        # Lagerreaktion berechnen
                        lager_reaktion_x, lager_reaktion_y = self.berechne_lagerreaktion_komponenten(balken, lager)
                        
                        # Nur Querkraft-Komponente erzeugt Biegemoment
                        normal_x = -dy / balken_laenge
                        normal_y = dx / balken_laenge
                        
                        querkraft_komponente = lager_reaktion_x * normal_x + lager_reaktion_y * normal_y
                        gesamt_moment += querkraft_komponente * hebelarm
            
            # Skalierung für bessere Darstellung
            return gesamt_moment / 100
        
        return moment
    
    def get_parameter_on_balken(self, point, balken):
        """Berechnet den Parameter t (0-1) für einen Punkt auf dem Balken"""
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        
        # Verwende Projektion statt nur X oder Y Koordinaten
        px = point[0] - balken.start[0]
        py = point[1] - balken.start[1]
        
        if dx*dx + dy*dy == 0:  # Balken hat keine Länge
            return 0
        
        # Projektion des Punktes auf die Balkenlinie
        t = (px * dx + py * dy) / (dx * dx + dy * dy)
        return max(0, min(1, t))  # Auf [0,1] begrenzen
    
    def berechne_lagerreaktion_komponenten(self, balken, lager):
        """Berechnet die Lagerreaktion in X- und Y-Komponenten - korrekte statische Berechnung"""
        if lager['typ'] == LagerTyp.GELENK:
            # Gelenke übertragen keine Reaktionskräfte (nur Verbindung)
            return (0, 0)
        
        # Sammle alle äußeren Kräfte am Balken
        gesamt_kraft_x = 0
        gesamt_kraft_y = 0
        
        # Punktlasten
        for last in balken.punktlasten:
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            
            # Fallback für alte Punktlasten
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            gesamt_kraft_x += kraft_x
            gesamt_kraft_y += kraft_y
        
        # Linienlasten als resultierende Kräfte
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            
            if balken_laenge > 0:
                laenge = (end_t - start_t) * balken_laenge
                resultierende = last['kraft'] * laenge / 100
                
                # Linienlast wirkt senkrecht zum Balken
                normal_x = -dy / balken_laenge
                normal_y = dx / balken_laenge
                
                gesamt_kraft_x += resultierende * normal_x
                gesamt_kraft_y += resultierende * normal_y
        
        # Lager, die Reaktionen aufnehmen können (ohne Gelenke)
        reaktions_lager = [l for l in balken.lager 
                          if l['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG]]
        
        if len(reaktions_lager) == 0:
            return (0, 0)
        
        elif len(reaktions_lager) == 1:
            # Ein Lager - trägt alle Kräfte (statisch bestimmt)
            if lager == reaktions_lager[0]:
                # Bei Einspannung: alle Reaktionen
                if lager['typ'] == LagerTyp.EINSPANNUNG:
                    return (-gesamt_kraft_x, -gesamt_kraft_y)
                # Bei Festlager: beide Komponenten
                elif lager['typ'] == LagerTyp.FESTLAGER:
                    return (-gesamt_kraft_x, -gesamt_kraft_y)
                # Bei Loslager: nur eine Komponente (vereinfacht: vertikal)
                elif lager['typ'] == LagerTyp.LOSLAGER:
                    return (0, -gesamt_kraft_y)
            return (0, 0)
        
        elif len(reaktions_lager) == 2:
            # Zwei Lager - statisch bestimmte Berechnung durch Momentengleichgewicht
            if lager in reaktions_lager:
                andere_lager = [l for l in reaktions_lager if l != lager][0]
                
                # Parameter der Lager entlang des Balkens
                lager_t = self.get_parameter_on_balken(lager['position'], balken)
                andere_t = self.get_parameter_on_balken(andere_lager['position'], balken)
                
                balken_laenge = math.sqrt((balken.end[0] - balken.start[0])**2 + 
                                        (balken.end[1] - balken.start[1])**2)
                
                # Momentengleichgewicht um das andere Lager
                moment_um_anderes = 0
                
                # Momente der Punktlasten
                for last in balken.punktlasten:
                    last_t = self.get_parameter_on_balken(last['position'], balken)
                    hebelarm = (last_t - andere_t) * balken_laenge
                    
                    kraft_x = last.get('kraft_x', 0)
                    kraft_y = last.get('kraft_y', 0)
                    if kraft_x == 0 and kraft_y == 0:
                        kraft_y = last.get('kraft', 0)
                    
                    # Nur Y-Komponente für vereinfachte Berechnung
                    moment_um_anderes += kraft_y * hebelarm
                
                # Momente der Linienlasten
                for last in balken.linienlasten:
                    start_t = self.get_parameter_on_balken(last['start'], balken)
                    end_t = self.get_parameter_on_balken(last['end'], balken)
                    schwerpunkt_t = (start_t + end_t) / 2
                    hebelarm = (schwerpunkt_t - andere_t) * balken_laenge
                    
                    laenge = (end_t - start_t) * balken_laenge
                    resultierende = last['kraft'] * laenge / 100
                    
                    moment_um_anderes += resultierende * hebelarm
                
                # Reaktion dieses Lagers aus Momentengleichgewicht
                lager_hebelarm = (lager_t - andere_t) * balken_laenge
                
                if abs(lager_hebelarm) > 1e-6:
                    reaktion_y = -moment_um_anderes / lager_hebelarm
                    
                    # X-Komponente aus Kräftegleichgewicht
                    if lager['typ'] == LagerTyp.FESTLAGER:
                        reaktion_x = -gesamt_kraft_x / 2  # Vereinfacht
                    else:
                        reaktion_x = 0
                    
                    return (reaktion_x, reaktion_y)
                
                # Fallback: gleichmäßige Verteilung
                return (-gesamt_kraft_x / 2, -gesamt_kraft_y / 2)
            
            return (0, 0)
        
        else:
            # Mehr als 2 Lager - statisch unbestimmt, gleichmäßige Verteilung
            if lager in reaktions_lager:
                return (-gesamt_kraft_x / len(reaktions_lager), -gesamt_kraft_y / len(reaktions_lager))
            return (0, 0)
    
    def berechne_einspannmoment_korrekt(self, balken, einspann_lager):
        """Berechnet das Einspannmoment korrekt durch Momentengleichgewicht"""
        dx = balken.end[0] - balken.start[0]
        dy = balken.end[1] - balken.start[1]
        balken_laenge = math.sqrt(dx**2 + dy**2)
        
        if balken_laenge == 0:
            return 0
        
        einspann_t = self.get_parameter_on_balken(einspann_lager['position'], balken)
        
        # Momentengleichgewicht um die Einspannung aufstellen
        gesamt_moment = 0
        
        # Momente aller Punktlasten um die Einspannung
        for last in balken.punktlasten:
            last_t = self.get_parameter_on_balken(last['position'], balken)
            hebelarm = abs(last_t - einspann_t) * balken_laenge
            
            kraft_x = last.get('kraft_x', 0)
            kraft_y = last.get('kraft_y', 0)
            
            # Fallback für alte Punktlasten
            if kraft_x == 0 and kraft_y == 0:
                kraft_y = last.get('kraft', 0)
            
            # Nur Querkraft-Komponente erzeugt Biegemoment
            normal_x = -dy / balken_laenge
            normal_y = dx / balken_laenge
            
            querkraft_komponente = kraft_x * normal_x + kraft_y * normal_y
            
            # Vorzeichen je nach Position relativ zur Einspannung
            if last_t > einspann_t:
                gesamt_moment += querkraft_komponente * hebelarm
            else:
                gesamt_moment -= querkraft_komponente * hebelarm
        
        # Momente aller Linienlasten um die Einspannung
        for last in balken.linienlasten:
            start_t = self.get_parameter_on_balken(last['start'], balken)
            end_t = self.get_parameter_on_balken(last['end'], balken)
            
            # Schwerpunkt der Linienlast
            schwerpunkt_t = (start_t + end_t) / 2
            hebelarm = abs(schwerpunkt_t - einspann_t) * balken_laenge
            
            # Resultierende Kraft der Linienlast
            linienlast_laenge = (end_t - start_t) * balken_laenge
            resultierende_kraft = last['kraft'] * linienlast_laenge / 100
            
            # Linienlast wirkt senkrecht zum Balken
            normal_y = dx / balken_laenge
            querkraft_komponente = resultierende_kraft * normal_y
            
            # Vorzeichen je nach Position relativ zur Einspannung
            if schwerpunkt_t > einspann_t:
                gesamt_moment += querkraft_komponente * hebelarm
            else:
                gesamt_moment -= querkraft_komponente * hebelarm
        
        # Das Einspannmoment muss das Gleichgewicht herstellen
        return -gesamt_moment

    def get_querkraft_funktion(self, balken):
        """Gibt eine Funktion für die Querkraft zurück"""
        def querkraft(x_rel):
            # Querkraft = Summe aller Kräfte rechts vom Schnitt
            gesamt_kraft = 0
            x_abs = balken.start[0] + x_rel * (balken.end[0] - balken.start[0])
            
            # Lagerreaktionen berücksichtigen (rechts vom Schnitt)
            for lager in balken.lager:
                lager_x = lager['position'][0]
                if lager_x >= x_abs:  # Lager rechts vom Schnitt
                    # Lagerreaktion berechnen (vereinfacht)
                    if lager['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG]:
                        lager_reaktion = self.berechne_lagerreaktion_vertikal(balken, lager)
                        gesamt_kraft += lager_reaktion
            
            # Punktlasten rechts von der Position
            for last in balken.punktlasten:
                last_x = last['position'][0]
                if last_x >= x_abs:  # Nur Lasten rechts vom Schnitt betrachten
                    kraft_y = last.get('kraft', 0)
                    gesamt_kraft += kraft_y
            
            # Globale Punktlasten rechts von der Position (nur die auf diesem Balken)
            for last in self.punktlasten_global:
                last_x = last['position'][0]
                last_y = last['position'][1]
                # Prüfe ob globale Punktlast auf diesem Balken liegt
                if (balken.point_on_beam((last_x, last_y), tolerance=15) and 
                    last_x >= x_abs):  # Nur Lasten rechts vom Schnitt betrachten
                    kraft_y = last.get('kraft_y', 0)
                    gesamt_kraft += kraft_y
            
            # Linienlasten rechts von der Position  
            for last in balken.linienlasten:
                start_x = last['start'][0]
                end_x = last['end'][0]
                
                # Bereich der Linienlast, der rechts vom Schnitt liegt
                schnitt_start = max(start_x, x_abs)
                schnitt_end = end_x
                
                if schnitt_start < schnitt_end:
                    # Resultierende der rechtsliegenden Linienlast
                    laenge_rechts = schnitt_end - schnitt_start
                    resultierende = last['kraft'] * laenge_rechts / 100  # Skalierung
                    gesamt_kraft += resultierende
            
            return gesamt_kraft  # Positiv für korrekte Darstellung
        
        return querkraft
    
    def berechne_lagerreaktion_vertikal(self, balken, lager):
        """Berechnet die vertikale Lagerreaktion (vereinfacht)"""
        # Vereinfachte Berechnung: Gleichgewicht der vertikalen Kräfte
        gesamt_last = 0
        
        # Alle vertikalen Lasten summieren
        for last in balken.punktlasten:
            kraft_y = last.get('kraft', 0)
            gesamt_last += kraft_y
        
        # Globale Punktlasten hinzufügen (nur die auf diesem Balken)
        for last in self.punktlasten_global:
            last_x = last['position'][0]
            last_y = last['position'][1]
            # Prüfe ob globale Punktlast auf diesem Balken liegt
            if balken.point_on_beam((last_x, last_y), tolerance=15):
                kraft_y = last.get('kraft_y', 0)
                gesamt_last += kraft_y
        
        for last in balken.linienlasten:
            start_x = last['start'][0]
            end_x = last['end'][0]
            kraft_pro_laenge = last['kraft']
            laenge = end_x - start_x
            resultierende = kraft_pro_laenge * laenge / 100
            gesamt_last += resultierende
        
        # Anzahl der Lager, die vertikale Reaktionen aufnehmen können
        vert_lager_count = sum(1 for l in balken.lager 
                              if l['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG])
        
        if vert_lager_count > 0:
            # Bei beidseitig fest gelagertem Balken: statisch bestimmte Berechnung
            if vert_lager_count == 2:
                # Vereinfachte Berechnung für 2 Lager: Momentengleichgewicht um das andere Lager
                andere_lager = [l for l in balken.lager if l != lager and 
                               l['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER, LagerTyp.EINSPANNUNG]]
                
                if andere_lager:
                    anderes_lager_x = andere_lager[0]['position'][0]
                    lager_x = lager['position'][0]
                    balken_laenge = abs(anderes_lager_x - lager_x)
                    
                    if balken_laenge > 0:
                        # Momentengleichgewicht um das andere Lager
                        summe_momente = 0
                        
                        # Momente von Punktlasten
                        for last in balken.punktlasten:
                            last_x = last['position'][0]
                            hebelarm = last_x - anderes_lager_x
                            kraft_y = last.get('kraft', 0)
                            summe_momente += kraft_y * hebelarm
                        
                        # Momente von globalen Punktlasten (nur die auf diesem Balken)
                        for last in self.punktlasten_global:
                            last_x = last['position'][0]
                            last_y = last['position'][1]
                            # Prüfe ob globale Punktlast auf diesem Balken liegt
                            if balken.point_on_beam((last_x, last_y), tolerance=15):
                                hebelarm = last_x - anderes_lager_x
                                kraft_y = last.get('kraft_y', 0)
                                summe_momente += kraft_y * hebelarm
                        
                        # Momente von Linienlasten
                        for last in balken.linienlasten:
                            start_x = last['start'][0]
                            end_x = last['end'][0]
                            schwerpunkt = (start_x + end_x) / 2
                            hebelarm = schwerpunkt - anderes_lager_x
                            kraft_pro_laenge = last['kraft']
                            laenge = end_x - start_x
                            resultierende = kraft_pro_laenge * laenge / 100
                            summe_momente += resultierende * hebelarm
                        
                        # Lagerreaktion aus Momentengleichgewicht
                        hebelarm_lager = lager_x - anderes_lager_x
                        if abs(hebelarm_lager) > 0.1:  # Vermeidung Division durch null
                            return -summe_momente / hebelarm_lager
            
            # Fallback: Gleichmäßige Verteilung
            return -gesamt_last / vert_lager_count
        
        return 0
    
    def get_normalkraft_funktion(self, balken):
        """Gibt eine Funktion für die Normalkraft zurück"""
        def normalkraft(x_rel):
            # Normalkraft = Summe aller axialen Kräfte (entlang der Balkenachse)
            gesamt_normalkraft = 0
            
            # Balken-Orientierung bestimmen
            dx = balken.end[0] - balken.start[0]
            dy = balken.end[1] - balken.start[1]
            balken_laenge = math.sqrt(dx**2 + dy**2)
            ist_horizontal = abs(dx) > abs(dy)
            
            if ist_horizontal:
                # Horizontaler Balken - normale x-basierte Berechnung
                x_abs = balken.start[0] + x_rel * dx
                
                if balken_laenge > 0:
                    # Normalisierte Richtungsvektoren
                    cos_alpha = dx / balken_laenge
                    sin_alpha = dy / balken_laenge
                    
                    # Punktlasten: Axiale Komponente berücksichtigen
                    for last in balken.punktlasten:
                        last_x = last['position'][0]
                        if last_x >= x_abs:  # Nur Lasten rechts vom Schnitt
                            kraft_y = last.get('kraft', 0)
                            
                            # Axiale Komponente = Fy*sin(α) (nur vertikale Kraft)
                            axiale_komponente = kraft_y * sin_alpha
                            gesamt_normalkraft += axiale_komponente
                    
                    # Globale Punktlasten: Axiale Komponente berücksichtigen (nur die auf diesem Balken)
                    for last in self.punktlasten_global:
                        last_x = last['position'][0]
                        last_y = last['position'][1]
                        # Prüfe ob globale Punktlast auf diesem Balken liegt
                        if (balken.point_on_beam((last_x, last_y), tolerance=15) and 
                            last_x >= x_abs):  # Nur Lasten rechts vom Schnitt
                            kraft_y = last.get('kraft_y', 0)
                            kraft_x = last.get('kraft_x', 0)
                            
                            # Axiale Komponente = Fx*cos(α) + Fy*sin(α)
                            axiale_komponente = kraft_x * cos_alpha + kraft_y * sin_alpha
                            gesamt_normalkraft += axiale_komponente
                    
                    # Linienlasten berücksichtigen
                    for last in balken.linienlasten:
                        start_x = last['start'][0]
                        end_x = last['end'][0]
                        
                        schnitt_start = max(start_x, x_abs)
                        schnitt_end = end_x
                        
                        if schnitt_start < schnitt_end:
                            laenge_rechts = schnitt_end - schnitt_start
                            resultierende = last['kraft'] * laenge_rechts / 100
                            axiale_komponente = resultierende * sin_alpha
                            gesamt_normalkraft += axiale_komponente
            else:
                # Vertikaler Balken - y-basierte Berechnung
                y_abs = balken.start[1] + x_rel * dy
                
                if balken_laenge > 0:
                    # Normalisierte Richtungsvektoren
                    cos_alpha = dx / balken_laenge
                    sin_alpha = dy / balken_laenge
                    
                    # Punktlasten: Axiale Komponente berücksichtigen
                    for last in balken.punktlasten:
                        last_y = last['position'][1]
                        if (dy > 0 and last_y >= y_abs) or (dy < 0 and last_y <= y_abs):
                            kraft_y = last.get('kraft', 0)
                            
                            # Axiale Komponente = Fy*sin(α) (nur vertikale Kraft)
                            axiale_komponente = kraft_y * sin_alpha
                            gesamt_normalkraft += axiale_komponente
                    
                    # Globale Punktlasten: Axiale Komponente berücksichtigen (nur die auf diesem Balken)
                    for last in self.punktlasten_global:
                        last_x = last['position'][0]
                        last_y = last['position'][1]
                        # Prüfe ob globale Punktlast auf diesem Balken liegt
                        if (balken.point_on_beam((last_x, last_y), tolerance=15) and 
                            ((dy > 0 and last_y >= y_abs) or (dy < 0 and last_y <= y_abs))):
                            kraft_y = last.get('kraft_y', 0)
                            kraft_x = last.get('kraft_x', 0)
                            
                            # Axiale Komponente = Fx*cos(α) + Fy*sin(α)
                            axiale_komponente = kraft_x * cos_alpha + kraft_y * sin_alpha
                            gesamt_normalkraft += axiale_komponente
            
            return gesamt_normalkraft
        
        return normalkraft
    
    def get_moment_funktion(self, balken):
        """Gibt eine Funktion für das Biegemoment zurück"""
        def moment(x_rel):
            x_abs = balken.start[0] + x_rel * (balken.end[0] - balken.start[0])
            gesamt_moment = 0
            
            # Prüfe Lagertypen - Festlager und Loslager können keine Momente aufnehmen
            festlager_count = sum(1 for l in balken.lager if l['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER])
            einspannung_count = sum(1 for l in balken.lager if l['typ'] == LagerTyp.EINSPANNUNG)
            
            # Nur Einspannungen können Momente aufnehmen
            for lager in balken.lager:
                lager_x = lager['position'][0]
                if lager['typ'] == LagerTyp.EINSPANNUNG:
                    # Einspannung kann Momente aufnehmen
                    if lager_x <= x_abs:  # Lager links vom Schnitt
                        einspann_moment = self.berechne_einspannmoment(balken, lager)
                        gesamt_moment -= einspann_moment / 100  # Skalierung
            
            # Lagerreaktionen von Fest-/Loslagern als Kräfte berücksichtigen
            for lager in balken.lager:
                lager_x = lager['position'][0]
                if lager['typ'] in [LagerTyp.FESTLAGER, LagerTyp.LOSLAGER] and lager_x >= x_abs:
                    # Lagerreaktion als Kraft rechts vom Schnitt
                    hebelarm = lager_x - x_abs
                    lager_reaktion = self.berechne_lagerreaktion_vertikal(balken, lager)
                    gesamt_moment += lager_reaktion * hebelarm / 100  # Skalierung
            
            # Momente von Punktlasten (rechts von der Schnittstelle)
            for last in balken.punktlasten:
                last_x = last['position'][0]
                if last_x >= x_abs:  # Nur Lasten rechts vom Schnitt betrachten
                    hebelarm = last_x - x_abs
                    kraft_y = last.get('kraft', 0)
                    gesamt_moment += kraft_y * hebelarm / 100  # Skalierung
            
            # Globale Punktlasten berücksichtigen (nur die auf diesem Balken)
            for last in self.punktlasten_global:
                last_x = last['position'][0]
                last_y = last['position'][1]
                # Prüfe ob globale Punktlast auf diesem Balken liegt
                if (balken.point_on_beam((last_x, last_y), tolerance=15) and 
                    last_x >= x_abs):  # Nur Lasten rechts vom Schnitt betrachten
                    hebelarm = last_x - x_abs
                    kraft_y = last.get('kraft_y', 0)
                    gesamt_moment += kraft_y * hebelarm / 100  # Skalierung
            
            # Momente von Linienlasten (rechts von der Schnittstelle)
            for last in balken.linienlasten:
                start_x = last['start'][0]
                end_x = last['end'][0]
                kraft_pro_laenge = last['kraft']
                
                # Bereich der Linienlast, der rechts vom Schnitt liegt
                schnitt_start = max(start_x, x_abs)
                schnitt_end = end_x
                
                if schnitt_start < schnitt_end:
                    # Schwerpunkt der rechtsliegenden Linienlast
                    schwerpunkt = (schnitt_start + schnitt_end) / 2
                    hebelarm = schwerpunkt - x_abs
                    
                    # Resultierende der rechtsliegenden Linienlast
                    laenge_rechts = schnitt_end - schnitt_start
                    resultierende = kraft_pro_laenge * laenge_rechts / 100  # Skalierung
                    
                    gesamt_moment += resultierende * hebelarm / 100
            
            return gesamt_moment
        
        return moment
    
    def berechne_einspannmoment(self, balken, einspann_lager):
        """Berechnet das Einspannmoment (vereinfacht)"""
        # Vereinfachte Berechnung: Summe aller Momente um die Einspannung
        einspann_x = einspann_lager['position'][0]
        gesamt_moment = 0
        
        # Momente von Punktlasten um die Einspannung
        for last in balken.punktlasten:
            last_x = last['position'][0]
            hebelarm = last_x - einspann_x
            kraft_y = last.get('kraft', 0)
            gesamt_moment += kraft_y * hebelarm
        
        # Momente von globalen Punktlasten um die Einspannung (nur die auf diesem Balken)
        for last in self.punktlasten_global:
            last_x = last['position'][0]
            last_y = last['position'][1]
            # Prüfe ob globale Punktlast auf diesem Balken liegt
            if balken.point_on_beam((last_x, last_y), tolerance=15):
                hebelarm = last_x - einspann_x
                kraft_y = last.get('kraft_y', 0)
                gesamt_moment += kraft_y * hebelarm
        
        # Momente von Linienlasten um die Einspannung
        for last in balken.linienlasten:
            start_x = last['start'][0]
            end_x = last['end'][0]
            kraft_pro_laenge = last['kraft']
            
            # Schwerpunkt der Linienlast
            schwerpunkt = (start_x + end_x) / 2
            hebelarm = schwerpunkt - einspann_x
            
            # Resultierende der Linienlast
            laenge = end_x - start_x
            resultierende = kraft_pro_laenge * laenge / 100
            
            gesamt_moment += resultierende * hebelarm
        
        return gesamt_moment
    
    def run(self):
        running = True
        while running:
            self.screen.fill((240, 240, 240))
            self.draw_grid()
            
            # Validiere Lager bei jedem Frame (für dynamische Korrekturen)
            self.validate_and_fix_lager()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    
                    # Slider Check (vor GUI Button Check)
                    if self.handle_slider_click(pos):
                        continue
                    
                    # GUI Button Check
                    if self.handle_button_click(pos):
                        continue
                    
                    # Je nach Modus handeln
                    if self.modus == Modus.BALKEN_ZEICHNEN:
                        if event.button == 1:  # Linke Maustaste
                            self.handle_balken_mode(pos)
                    elif self.modus == Modus.LAGER_SETZEN:
                        if event.button == 1:  # Linke Maustaste
                            self.handle_lager_mode(pos)
                    elif self.modus == Modus.PUNKTLASTEN_SETZEN:
                        if event.button == 1:  # Nur linke Maustaste
                            self.handle_punktlast_mode_click(pos, event.button)
                    elif self.modus == Modus.LINIENLASTEN_SETZEN:
                        if event.button == 1:  # Linke Maustaste
                            self.handle_linienlast_mode(pos)
                    elif self.modus == Modus.BEARBEITEN:
                        if event.button == 1:  # Linke Maustaste
                            self.handle_bearbeiten_mode(pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    
                    # Slider Release
                    if self.scale_slider['dragging']:
                        self.scale_slider['dragging'] = False
                    
                    if self.modus == Modus.PUNKTLASTEN_SETZEN and event.button == 1:
                        self.handle_punktlast_mode_release(pos)
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.scale_slider['dragging']:
                        self.handle_slider_drag(pygame.mouse.get_pos())
                
                elif event.type == pygame.KEYDOWN:
                    # Element löschen mit Entfernen-Taste
                    if event.key == pygame.K_DELETE:
                        if self.selected_balken:
                            self.delete_selected_element()
                    
                    # ESC zum Abbrechen
                    elif event.key == pygame.K_ESCAPE:
                        self.reset_linienlast_state()
                        if self.selected_balken:
                            self.selected_balken.selected = False
                            self.selected_balken.selected_lager = None
                            self.selected_balken.selected_punktlast = None
                            self.selected_balken.selected_linienlast = None
                            self.selected_balken = None
            
            # Zeichne alle Balken
            for balken in self.balken_liste:
                balken.draw(self.screen)
            
            # Automatische Berechnung für jeden Balken einzeln
            self.berechne_schnittkraefte()
            
            # Vorschau beim Balken zeichnen
            if self.modus == Modus.BALKEN_ZEICHNEN and self.startpunkt:
                maus_pos = pygame.mouse.get_pos()
                snapped_pos = self.snap_to_grid(maus_pos)
                
                # Beliebige Richtung (auch diagonal)
                dx = snapped_pos[0] - self.startpunkt[0]
                dy = snapped_pos[1] - self.startpunkt[1]
                laenge = math.sqrt(dx**2 + dy**2)
                
                if laenge > 20:  # Nur zeichnen wenn Mindestlänge erreicht
                    pygame.draw.line(self.screen, (100, 100, 255), self.startpunkt, snapped_pos, 2)
                    
                    # Länge anzeigen
                    font = pygame.font.Font(None, 20)
                    text = font.render(f"{int(laenge)}px", True, (100, 100, 255))
                    mid_x = (self.startpunkt[0] + snapped_pos[0]) // 2
                    mid_y = (self.startpunkt[1] + snapped_pos[1]) // 2
                    self.screen.blit(text, (mid_x + 5, mid_y - 10))
            
            # Vorschau beim Punktlast ziehen
            if self.dragging_punktlast and self.punktlast_start:
                maus_pos = pygame.mouse.get_pos()
                snapped_pos = self.snap_to_grid(maus_pos)
                
                # Beliebige Richtung - zeige Pfeil
                dx = snapped_pos[0] - self.punktlast_start[0]
                dy = snapped_pos[1] - self.punktlast_start[1]
                kraft_betrag = math.sqrt(dx**2 + dy**2)
                
                if kraft_betrag > 5:
                    # Farbe basierend auf Richtung
                    if abs(dx) > abs(dy):
                        # Überwiegend horizontal
                        color = (255, 100, 0) if dx > 0 else (0, 100, 255)
                    else:
                        # Überwiegend vertikal
                        color = (255, 0, 0) if dy > 0 else (0, 255, 0)
                    
                    pygame.draw.line(self.screen, color, self.punktlast_start, snapped_pos, 3)
                    
                    # Pfeilspitze zur besseren Richtungsanzeige
                    pfeil_laenge = 10
                    winkel = math.atan2(dy, dx)
                    
                    spitze_x1 = snapped_pos[0] - pfeil_laenge * math.cos(winkel - 0.5)
                    spitze_y1 = snapped_pos[1] - pfeil_laenge * math.sin(winkel - 0.5)
                    spitze_x2 = snapped_pos[0] - pfeil_laenge * math.cos(winkel + 0.5)
                    spitze_y2 = snapped_pos[1] - pfeil_laenge * math.sin(winkel + 0.5)
                    
                    pygame.draw.polygon(self.screen, color, [
                        snapped_pos, (spitze_x1, spitze_y1), (spitze_x2, spitze_y2)
                    ])
                    
                    # Kraftwert anzeigen
                    font = pygame.font.Font(None, 24)
                    text = font.render(f"{int(kraft_betrag)}N", True, color)
                    self.screen.blit(text, (snapped_pos[0] + 10, snapped_pos[1] - 10))
            
            # Vorschau für Linienlast
            if self.modus == Modus.LINIENLASTEN_SETZEN:
                maus_pos = pygame.mouse.get_pos()
                snapped_pos = self.snap_to_grid(maus_pos)
                
                if self.linienlast_state == 1 and self.linienlast_start and self.linienlast_balken:
                    # Zeige mögliche Endposition auf dem Balken
                    end_preview = self.project_point_on_beam(snapped_pos, self.linienlast_balken)
                    pygame.draw.line(self.screen, (0, 255, 255), self.linienlast_start, end_preview, 3)
                
                elif self.linienlast_state == 2 and self.linienlast_start and self.linienlast_end and self.linienlast_balken:
                    # Berechne Kraft-Projektion senkrecht zum Balken
                    balken = self.linienlast_balken
                    dx = balken.end[0] - balken.start[0]
                    dy = balken.end[1] - balken.start[1]
                    balken_laenge = math.sqrt(dx**2 + dy**2)
                    
                    if balken_laenge > 0:
                        # Normierter Normalvektor (senkrecht zum Balken)
                        normal_x = -dy / balken_laenge
                        normal_y = dx / balken_laenge
                        
                        # Vektor vom Linienlast-Start zum Mauszeiger
                        start_to_mouse_x = snapped_pos[0] - self.linienlast_start[0]
                        start_to_mouse_y = snapped_pos[1] - self.linienlast_start[1]
                        
                        # Projektion auf die Normale = Kraftbetrag
                        kraft = start_to_mouse_x * normal_x + start_to_mouse_y * normal_y
                        
                        # Zeige Kraftvektor von der Mitte der Linienlast
                        mid_x = (self.linienlast_start[0] + self.linienlast_end[0]) // 2
                        mid_y = (self.linienlast_start[1] + self.linienlast_end[1]) // 2
                        kraft_end_x = mid_x + kraft * normal_x
                        kraft_end_y = mid_y + kraft * normal_y
                        
                        color = (255, 0, 0) if kraft < 0 else (0, 255, 0)
                        
                        # Rechteck zwischen Balken und Kraftvektor
                        points = [
                            self.linienlast_start,
                            self.linienlast_end,
                            (self.linienlast_end[0] + kraft * normal_x, self.linienlast_end[1] + kraft * normal_y),
                            (self.linienlast_start[0] + kraft * normal_x, self.linienlast_start[1] + kraft * normal_y)
                        ]
                        pygame.draw.polygon(self.screen, (*color[:3], 50), points)
                        pygame.draw.polygon(self.screen, color, points, 2)
                        
                        # Zeige den Kraftvektor
                        pygame.draw.line(self.screen, color, (mid_x, mid_y), (kraft_end_x, kraft_end_y), 4)
                        
                        # Pfeilspitze
                        if abs(kraft) > 5:
                            pfeil_laenge = 8
                            winkel = math.atan2(kraft * normal_y, kraft * normal_x)
                            spitze_x1 = kraft_end_x - pfeil_laenge * math.cos(winkel - 0.5)
                            spitze_y1 = kraft_end_y - pfeil_laenge * math.sin(winkel - 0.5)
                            spitze_x2 = kraft_end_x - pfeil_laenge * math.cos(winkel + 0.5)
                            spitze_y2 = kraft_end_y - pfeil_laenge * math.sin(winkel + 0.5)
                            
                            pygame.draw.polygon(self.screen, color, [
                                (kraft_end_x, kraft_end_y),
                                (spitze_x1, spitze_y1),
                                (spitze_x2, spitze_y2)
                            ])
                        
                        # Kraftwert anzeigen
                        font = pygame.font.Font(None, 24)
                        text = font.render(f"{abs(int(kraft))}N/m", True, color)
                        self.screen.blit(text, (kraft_end_x + 10, kraft_end_y - 10))
            
            # GUI zeichnen
            self.draw_gui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

# Hauptprogramm starten
if __name__ == "__main__":
    tool = SchnittkraftTool()
    tool.run()