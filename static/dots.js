const canvas = document.getElementById('dotsCanvas');
const ctx = canvas.getContext('2d');

let width, height;
let dots = [];

// Configuration
const spacing = 30; // distance between dots
const dotRadius = 1.5;
const maxDistance = 150; // mouse interaction radius
const repulsionForce = 15; // how much dots move away

let mouse = {
    x: null,
    y: null
};

function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
    initDots();
}

class Dot {
    constructor(x, y) {
        this.baseX = x;
        this.baseY = y;
        this.x = x;
        this.y = y;
    }

    update() {
        if (mouse.x === null || mouse.y === null) return;

        const dx = mouse.x - this.baseX;
        const dy = mouse.y - this.baseY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < maxDistance) {
            const force = (maxDistance - distance) / maxDistance;
            const angle = Math.atan2(dy, dx);
            
            // Move away from mouse
            const targetX = this.baseX - Math.cos(angle) * force * repulsionForce;
            const targetY = this.baseY - Math.sin(angle) * force * repulsionForce;

            this.x += (targetX - this.x) * 0.1;
            this.y += (targetY - this.y) * 0.1;
        } else {
            // Return to base position
            this.x += (this.baseX - this.x) * 0.05;
            this.y += (this.baseY - this.y) * 0.05;
        }
    }

    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, dotRadius, 0, Math.PI * 2);
        // Use a subtle color for the dots, maybe picking up the theme slightly
        // We will make it slightly transparent gray or theme accent based
        ctx.fillStyle = 'rgba(150, 150, 150, 0.4)'; 
        ctx.fill();
    }
}

function initDots() {
    dots = [];
    for (let y = spacing / 2; y < height; y += spacing) {
        for (let x = spacing / 2; x < width; x += spacing) {
            dots.push(new Dot(x, y));
        }
    }
}

function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < dots.length; i++) {
        dots[i].update();
        dots[i].draw();
    }

    requestAnimationFrame(animate);
}

// Event Listeners
window.addEventListener('resize', resize);

window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});

window.addEventListener('mouseleave', () => {
    mouse.x = null;
    mouse.y = null;
});

// Initialize
resize();
animate();
