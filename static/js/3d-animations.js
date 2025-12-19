

class ModernAnimations {
    constructor() {
        this.init();
    }

    init() {
        this.setupScrollAnimations();
        this.setupParticles();
        this.setupMouseEffects();
        this.setupCardInteractions();
        this.setupCounterAnimations();
        this.setupFormAnimations();
    }

    
    setupScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        
        document.querySelectorAll('.fade-in-up, .fade-in-left, .fade-in-right').forEach(el => {
            observer.observe(el);
        });
    }

    
    setupParticles() {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particles';
        document.body.appendChild(particleContainer);

        
        for (let i = 0; i < 50; i++) {
            setTimeout(() => {
                this.createParticle(particleContainer);
            }, i * 300);
        }

        
        setInterval(() => {
            this.createParticle(particleContainer);
        }, 3000);
    }

    createParticle(container) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particle.style.animationDelay = Math.random() * 2 + 's';
        
        
        const colors = [
            'rgba(102, 126, 234, 0.6)',
            'rgba(118, 75, 162, 0.6)',
            'rgba(240, 147, 251, 0.6)',
            'rgba(79, 172, 254, 0.6)'
        ];
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];
        
        container.appendChild(particle);
        
        
        setTimeout(() => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
            }
        }, 15000);
    }

    
    setupMouseEffects() {
        let mouseX = 0;
        let mouseY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;

            
            document.querySelectorAll('.float-element').forEach((el, index) => {
                const speed = (index + 1) * 0.02;
                const x = (mouseX - window.innerWidth / 2) * speed;
                const y = (mouseY - window.innerHeight / 2) * speed;
                
                el.style.transform = `translate(${x}px, ${y}px)`;
            });
        });

        
        document.querySelectorAll('.btn-3d').forEach(btn => {
            btn.addEventListener('mousemove', (e) => {
                const rect = btn.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                
                btn.style.transform = `translate(${x * 0.1}px, ${y * 0.1}px) scale(1.05)`;
            });

            btn.addEventListener('mouseleave', () => {
                btn.style.transform = 'translate(0px, 0px) scale(1)';
            });
        });
    }

    
    setupCardInteractions() {
        document.querySelectorAll('.card-3d').forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = (y - centerY) / 10;
                const rotateY = (centerX - x) / 10;
                
                card.style.transform = `
                    translateY(-10px) 
                    rotateX(${rotateX}deg) 
                    rotateY(${rotateY}deg)
                    scale(1.02)
                `;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) rotateX(0) rotateY(0) scale(1)';
            });
        });
    }

    
    setupCounterAnimations() {
        const animateCounter = (element, target, duration = 2000) => {
            let start = 0;
            const increment = target / (duration / 16);
            
            const timer = setInterval(() => {
                start += increment;
                element.textContent = Math.floor(start);
                
                if (start >= target) {
                    element.textContent = target;
                    clearInterval(timer);
                }
            }, 16);
        };

        
        const counterObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = parseInt(entry.target.dataset.target) || 
                                 parseInt(entry.target.textContent) || 0;
                    animateCounter(entry.target, target);
                    counterObserver.unobserve(entry.target);
                }
            });
        });

        document.querySelectorAll('.stats-number-3d, .stats-number').forEach(el => {
            counterObserver.observe(el);
        });
    }

    
    setupFormAnimations() {
        document.querySelectorAll('.form-control-3d').forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('focused');
            });

            input.addEventListener('blur', () => {
                if (!input.value) {
                    input.parentElement.classList.remove('focused');
                }
            });

            
            if (input.value) {
                input.parentElement.classList.add('focused');
            }
        });
    }

    
    smoothScrollTo(target, duration = 1000) {
        const targetElement = document.querySelector(target);
        if (!targetElement) return;

        const targetPosition = targetElement.offsetTop;
        const startPosition = window.pageYOffset;
        const distance = targetPosition - startPosition;
        let startTime = null;

        const animation = (currentTime) => {
            if (startTime === null) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const run = this.easeInOutQuart(timeElapsed, startPosition, distance, duration);
            window.scrollTo(0, run);
            if (timeElapsed < duration) requestAnimationFrame(animation);
        };

        requestAnimationFrame(animation);
    }

    
    easeInOutQuart(t, b, c, d) {
        t /= d / 2;
        if (t < 1) return c / 2 * t * t * t * t + b;
        t -= 2;
        return -c / 2 * (t * t * t * t - 2) + b;
    }

    
    createRipple(event) {
        const button = event.currentTarget;
        const circle = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;

        circle.style.width = circle.style.height = `${diameter}px`;
        circle.style.left = `${event.clientX - button.offsetLeft - radius}px`;
        circle.style.top = `${event.clientY - button.offsetTop - radius}px`;
        circle.classList.add('ripple');

        const ripple = button.getElementsByClassName('ripple')[0];
        if (ripple) {
            ripple.remove();
        }

        button.appendChild(circle);
    }

    
    staggerAnimation(elements, delay = 100) {
        elements.forEach((el, index) => {
            setTimeout(() => {
                el.classList.add('animate');
            }, index * delay);
        });
    }

    
    createMorphingLoader(container) {
        const loader = document.createElement('div');
        loader.className = 'morphing-loader';
        loader.innerHTML = `
            <div class="morph-shape"></div>
            <div class="morph-shape"></div>
            <div class="morph-shape"></div>
        `;
        container.appendChild(loader);
        return loader;
    }

    
    removeLoader(loader) {
        if (loader && loader.parentNode) {
            loader.style.opacity = '0';
            setTimeout(() => {
                loader.parentNode.removeChild(loader);
            }, 300);
        }
    }
}


const additionalCSS = `
.ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0);
    animation: ripple-animation 0.6s linear;
    pointer-events: none;
}

@keyframes ripple-animation {
    to {
        transform: scale(4);
        opacity: 0;
    }
}

.morphing-loader {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
}

.morph-shape {
    width: 12px;
    height: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 50%;
    animation: morph 1.5s ease-in-out infinite;
}

.morph-shape:nth-child(2) {
    animation-delay: 0.2s;
}

.morph-shape:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes morph {
    0%, 100% {
        transform: scale(1) rotate(0deg);
        border-radius: 50%;
    }
    50% {
        transform: scale(1.2) rotate(180deg);
        border-radius: 20%;
    }
}

.focused .form-label-3d {
    color: #667eea;
    transform: translateY(-5px) scale(0.9);
}
`;


const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);


document.addEventListener('DOMContentLoaded', () => {
    new ModernAnimations();
});


document.addEventListener('click', (e) => {
    if (e.target.classList.contains('btn-3d')) {
        const animations = new ModernAnimations();
        animations.createRipple(e);
    }
});


window.ModernAnimations = ModernAnimations;