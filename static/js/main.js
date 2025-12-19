


let dashboardDataCache = null;
let cacheTimestamp = null;
const CACHE_DURATION = 5 * 60 * 1000; 


function showLoading(element) {
    const loadingHTML = `
        <div class="text-center py-3">
            <div class="spinner-border spinner-border-sm text-primary" role="status">
                <span class="visually-hidden">جاري التحميل...</span>
            </div>
            <p class="mt-2 text-muted small">جاري تحميل البيانات الحقيقية...</p>
        </div>
    `;
    $(element).html(loadingHTML);
}


function showSuccess(element, message) {
    $(element).html(`
        <div class="text-center py-4">
            <i class="fas fa-check-circle text-success" style="font-size: 3rem;"></i>
            <p class="mt-2 text-success">${message}</p>
        </div>
    `);
}


function showError(element, message) {
    $(element).html(`
        <div class="text-center py-4">
            <i class="fas fa-exclamation-triangle text-danger" style="font-size: 3rem;"></i>
            <p class="mt-2 text-danger">${message}</p>
        </div>
    `);
}


function scrollToElement(element) {
    $('html, body').animate({
        scrollTop: $(element).offset().top - 100
    }, 500);
}


function formatNumber(number) {
    return new Intl.NumberFormat('ar-EG').format(number);
}


function animateCounter(element, target) {
    const $element = $(element);
    const start = 0;
    const duration = 800; 
    const startTime = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + (target - start) * easeOut);
        
        $element.text(current);
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            $element.text(target);
        }
    }
    
    requestAnimationFrame(updateCounter);
}


function getDashboardData() {
    return new Promise((resolve, reject) => {
        const now = Date.now();
        
        
        if (dashboardDataCache && cacheTimestamp && (now - cacheTimestamp < CACHE_DURATION)) {
            console.log('Using cached dashboard data');
            resolve(dashboardDataCache);
            return;
        }
        
        
        $.get('/api/data/')
            .done(function(data) {
                console.log('Fresh dashboard data loaded:', data);
                dashboardDataCache = data;
                cacheTimestamp = now;
                resolve(data);
            })
            .fail(function(xhr, status, error) {
                console.error('Failed to load dashboard data:', error);
                reject(error);
            });
    });
}


function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    
    document.querySelectorAll('.feature-card, .stats-card, .card').forEach(card => {
        observer.observe(card);
    });
}


function ensureLayoutConsistency() {
    
    $('.main-content .row').each(function() {
        if (!$(this).hasClass('mb-4') && !$(this).hasClass('mb-5')) {
            $(this).addClass('mb-4');
        }
    });
    
    
    $('.row').each(function() {
        const cards = $(this).find('.card');
        if (cards.length > 1) {
            let maxHeight = 0;
            cards.each(function() {
                const height = $(this).outerHeight();
                if (height > maxHeight) maxHeight = height;
            });
            cards.css('min-height', maxHeight + 'px');
        }
    });
}


$(document).ready(function() {
    
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    
    ensureLayoutConsistency();
    
    
    initScrollAnimations();
    
    
    $('.card').hover(
        function() {
            $(this).addClass('shadow-lg');
        },
        function() {
            $(this).removeClass('shadow-lg');
        }
    );
    
    
    $('.feature-card').mousemove(function(e) {
        const card = $(this);
        const rect = this.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const rotateX = (y - centerY) / 10;
        const rotateY = (centerX - x) / 10;
        
        card.css('transform', `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(0)`);
    });
    
    $('.feature-card').mouseleave(function() {
        $(this).css('transform', 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0)');
    });
});