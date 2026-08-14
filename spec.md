# Dermatoscan Pro AI: UI/UX & Architecture Design Specification v2.0

## Overview
This document outlines the drastically evolved design system for the **Dermatoscan Pro AI** web interface. Drawing heavy inspiration from Advanced Bento Box layouts, Cyber-Tactile hardware interfaces, and multi-layered Glassmorphism, this specification defines a highly complex, visually stunning, and densely informative medical application interface.

## 1. Aesthetic Fusion: "Advanced Cyber-Tactile Bento"
The design language completely abandons standard web layouts (like simple side-by-side columns). It embraces the complexity of high-end hardware, audio synthesizers, and futuristic dashboards.

### Core Inspirations:
*   **Complex Bento Box Grids:** Layouts must use irregular, densely packed grid systems (Bento boxes) that overlap or interlock, rather than standard flex rows.
*   **Cyber-Tactile Hardware:** Toggles, sliders, and buttons must mimic physical, high-fidelity hardware. Think of rotary knobs, deep inset sliders, and physical switches with LED indicators.
*   **Neumorphism + Glassmorphism Hybrid:** Floating frosted glass panels layered directly over deep, soft Neumorphic (extruded/inset) base layers. 
*   **High-Tech Dark Mode:** The UI must feel like a specialized, professional-grade diagnostic instrument in a dark room.

## 2. Color Palette & Theming
*   **Background:** Deep Space Obsidian (`#0A0A0E`) to Heavy Carbon (`#121317`).
*   **Ambient Glow:** Extremely subtle, highly blurred background shapes (neon green or cyan) to provide backlight to the glass panels.
*   **Accents:**
    *   **Primary:** Neon Cyber-Green (`#00FF66`) or Electric Cyan (`#00F0FF`). Used for active hardware LEDs, data visualization, and typography highlights.
    *   **Secondary:** Deep Charcoal and Gunmetal for the hardware surfaces.

## 3. Topography & Surfaces
1.  **Level 0 (Base Plate):** A soft, Neumorphic dark base that looks like a solid piece of machined metal or matte plastic.
2.  **Level 1 (The Grid):** Bento-style containers that are either extruded out (soft drop shadow and upper highlight) or carved in (deep inset shadows).
3.  **Level 2 (Hardware Controls):** Highly realistic tactile elements. Sliders in deep tracks, rotary dials with tick marks, glowing LED toggles.
4.  **Level 3 (Data Glass):** High-blur `backdrop-filter` glass panels floating above the bento grid, used for displaying the AI image analysis and charts.

## 4. Typography
*   **Primary Font:** `Outfit` or `Space Grotesk`. Geometric, tech-focused, and sharp. 
*   **Data Readouts:** A monospace font (e.g., `JetBrains Mono` or `Fira Code`) must be used for all scores, metrics, and technical readouts to look like a terminal or HUD.

## 5. Layout Architecture (The Bento Grid)
*   **No More Basic Side-by-Side:** The layout must be a CSS Grid (`display: grid`) with varying row and column spans (`grid-column: span X`).
*   **Density:** The dashboard should feel dense but organized.
*   **Widgets:** 
    *   The Lesion Scanner should be a dominant central glass panel.
    *   ABCDE metrics should be small, interlocking bento boxes surrounding the scanner.
    *   Hardware toggles (Hair Removal, TTA) should be physical-looking switches in a dedicated control panel bento box.

## 6. Architecture Re-Evaluation (Backend)
*   The backend architecture must be re-roasted to ensure it supports this highly dense, multi-widget frontend without bottlenecking.
*   The architecture must ensure flawless real-time responsiveness for the cyber-tactile UI elements.

## 7. Actionable Implementation Goals
1.  **CSS Grid Rewrite:** Delete the old flexbox `.pane-grid` and implement an asymmetrical CSS Bento Grid.
2.  **Hardware UI:** Rebuild toggles and buttons to look like physical, 3D equipment with glowing LEDs.
3.  **Glass on Neumorphism:** Combine heavy inset/outset shadows on the bento boxes with translucent glass overlays for the data.
