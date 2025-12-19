
function getDashboardData() {
    return $.ajax({
        url: '/dashboard/api/data/',
        method: 'GET',
        dataType: 'json',
        timeout: 15000, 
        cache: true,
        beforeSend: function() {
            console.log('🔄 بدء تحميل البيانات من الخادم...');
        }
    });
}


function animateCounter(element, target, duration = 2000) {
    const $element = $(element);
    const start = 0;
    const startTime = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = Math.floor(start + (target - start) * easeOutQuart);
        
        $element.text(current);
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            $element.text(target); 
        }
    }
    
    requestAnimationFrame(updateCounter);
}


function updateProgressBars(clusterData) {
    const total = clusterData.reduce((sum, c) => sum + c.count, 0);
    
    clusterData.forEach((cluster, index) => {
        const percentage = Math.round((cluster.count / total) * 100);
        const progressBarIds = ['excellentProgressBar', 'goodProgressBar', 'averageProgressBar', 'supportProgressBar'];
        const percentageIds = ['excellentPercentage', 'goodPercentage', 'averagePercentage', 'supportPercentage'];
        
        if (progressBarIds[index] && percentageIds[index]) {
            
            setTimeout(() => {
                $(`#${progressBarIds[index]}`).css('width', percentage + '%');
                $(`#${percentageIds[index]}`).text(percentage + '%');
            }, index * 200);
        }
    });
}


function createChart(clusterData, type = 'bar') {
    console.log('🎨 إنشاء رسم بياني:', type, clusterData);
    
    if (!clusterData || clusterData.length === 0) {
        console.error('❌ لا توجد بيانات للرسم البياني');
        return;
    }
    
    let trace, layout;
    
    
    const gradientColors = [
        'rgba(40, 167, 69, 0.8)',   
        'rgba(23, 162, 184, 0.8)',  
        'rgba(255, 193, 7, 0.8)',   
        'rgba(220, 53, 69, 0.8)'    
    ];
    
    const solidColors = [
        '#28a745', '#17a2b8', '#ffc107', '#dc3545'
    ];
    
    if (type === 'bar') {
        trace = {
            x: clusterData.map(c => c.name),
            y: clusterData.map(c => c.count),
            type: 'bar',
            marker: {
                color: gradientColors,
                line: { 
                    color: '#FFFFFF', 
                    width: 3 
                },
                opacity: 0.9,
                
                pattern: {
                    shape: '/',
                    bgcolor: 'rgba(255,255,255,0.1)',
                    fgcolor: 'rgba(255,255,255,0.2)',
                    size: 8,
                    solidity: 0.3
                }
            },
            text: clusterData.map(c => `${c.count} طالب`),
            textposition: 'auto',
            textfont: { 
                color: 'white', 
                size: 14, 
                family: 'Cairo', 
                weight: 'bold' 
            },
            customdata: clusterData.map(c => c.description || 'وصف المجموعة'),
            hovertemplate: '<b>%{x}</b><br>' +
                          'عدد الطلاب: %{y}<br>' +
                          'النسبة: %{customdata}<br>' +
                          '<extra></extra>',
            hoverlabel: {
                bgcolor: solidColors,
                bordercolor: '#FFFFFF',
                font: { family: 'Cairo', size: 13, color: 'white' }
            }
        };
        
        layout = {
            title: {
                text: '📊 توزيع الطلاب حسب مستوى الأداء',
                font: { 
                    family: 'Cairo', 
                    size: 18, 
                    color: '#2C2C2C',
                    weight: 'bold'
                },
                x: 0.5,
                y: 0.95
            },
            xaxis: { 
                title: { 
                    text: 'مجموعات الطلاب', 
                    font: { family: 'Cairo', size: 14, color: '#2C2C2C' } 
                },
                font: { family: 'Cairo', size: 12, color: '#4A4A4A' },
                tickangle: -15,
                showgrid: false,
                linecolor: '#E0D5C7',
                linewidth: 2,
                tickcolor: '#E0D5C7'
            },
            yaxis: { 
                title: { 
                    text: 'عدد الطلاب', 
                    font: { family: 'Cairo', size: 14, color: '#2C2C2C' } 
                },
                gridcolor: '#F5F1ED',
                gridwidth: 1,
                font: { family: 'Cairo', size: 12, color: '#4A4A4A' },
                linecolor: '#E0D5C7',
                linewidth: 2,
                zeroline: false,
                tickcolor: '#E0D5C7'
            },
            bargap: 0.3,
            bargroupgap: 0.1
        };
    } else if (type === 'pie') {
        trace = {
            labels: clusterData.map(c => c.name),
            values: clusterData.map(c => c.count),
            type: 'pie',
            marker: {
                colors: solidColors,
                line: { 
                    color: '#FFFFFF', 
                    width: 4 
                }
            },
            textinfo: 'label+percent+value',
            textfont: { 
                family: 'Cairo', 
                size: 12, 
                weight: 'bold', 
                color: 'white' 
            },
            textposition: 'inside',
            customdata: clusterData.map(c => c.description || 'وصف المجموعة'),
            hovertemplate: '<b>%{label}</b><br>' +
                          'العدد: %{value}<br>' +
                          'النسبة: %{percent}<br>' +
                          'الوصف: %{customdata}<br>' +
                          '<extra></extra>',
            hoverlabel: {
                bgcolor: solidColors,
                bordercolor: '#FFFFFF',
                font: { family: 'Cairo', size: 14, color: 'white' }
            },
            
            pull: [0.05, 0.05, 0.05, 0.1] 
        };
        
        layout = {
            title: {
                text: '🥧 التوزيع الدائري للطلاب',
                font: { 
                    family: 'Cairo', 
                    size: 18, 
                    color: '#2C2C2C',
                    weight: 'bold'
                },
                x: 0.5,
                y: 0.95
            }
        };
    } else if (type === 'donut') {
        const total = clusterData.reduce((sum, c) => sum + c.count, 0);
        
        trace = {
            labels: clusterData.map(c => c.name),
            values: clusterData.map(c => c.count),
            type: 'pie',
            hole: 0.5,
            marker: {
                colors: solidColors,
                line: { 
                    color: '#FFFFFF', 
                    width: 4 
                }
            },
            textinfo: 'label+percent',
            textfont: { 
                family: 'Cairo', 
                size: 11, 
                weight: 'bold', 
                color: 'white' 
            },
            textposition: 'inside',
            customdata: clusterData.map(c => c.description || 'وصف المجموعة'),
            hovertemplate: '<b>%{label}</b><br>' +
                          'العدد: %{value}<br>' +
                          'النسبة: %{percent}<br>' +
                          'الوصف: %{customdata}<br>' +
                          '<extra></extra>',
            hoverlabel: {
                bgcolor: solidColors,
                bordercolor: '#FFFFFF',
                font: { family: 'Cairo', size: 14, color: 'white' }
            }
        };
        
        layout = {
            title: {
                text: '🍩 الرسم الحلقي للطلاب',
                font: { 
                    family: 'Cairo', 
                    size: 18, 
                    color: '#2C2C2C',
                    weight: 'bold'
                },
                x: 0.5,
                y: 0.95
            },
            annotations: [{
                text: `إجمالي الطلاب<br><b style="font-size: 24px; color: #B85C57;">${total}</b><br><span style="font-size: 12px; color: #6B6B6B;">طالب وطالبة</span>`,
                x: 0.5, 
                y: 0.5,
                font: { 
                    family: 'Cairo', 
                    size: 16, 
                    weight: 'bold', 
                    color: '#2C2C2C' 
                },
                showarrow: false,
                align: 'center'
            }]
        };
    }
    
    const commonLayout = {
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 60, b: 40, l: 40, r: 40 },
        showlegend: type !== 'bar',
        legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.1,
            font: { 
                family: 'Cairo', 
                size: 12,
                color: '#2C2C2C'
            },
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E0D5C7',
            borderwidth: 1
        },
        font: { family: 'Cairo' },
        hovermode: 'closest',
        dragmode: false,
        
        transition: {
            duration: 500,
            easing: 'cubic-in-out'
        }
    };
    
    Object.assign(layout, commonLayout);
    
    try {
        
        Plotly.newPlot('clusterChart', [trace], layout, {
            responsive: true,
            displayModeBar: false,
            staticPlot: false,
            scrollZoom: false,
            doubleClick: false,
            showTips: false,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
        });
        
        console.log('✅ تم إنشاء الرسم البياني بنجاح');
        
        
        document.getElementById('clusterChart').on('plotly_hover', function(data) {
            console.log('🎯 تم التمرير على:', data.points[0]);
        });
        
    } catch (error) {
        console.error('❌ خطأ في إنشاء الرسم البياني:', error);
        
        
        document.getElementById('clusterChart').innerHTML = `
            <div class="alert alert-danger text-center" style="margin: 20px;">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>خطأ في عرض الرسم البياني</strong><br>
                <small>يرجى المحاولة مرة أخرى أو تحديث الصفحة</small>
            </div>
        `;
    }
}


function showStatusMessage(message, type = 'info') {
    const alertClass = type === 'success' ? 'alert-success' : 
                      type === 'warning' ? 'alert-warning' : 
                      type === 'error' ? 'alert-danger' : 'alert-info';
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'warning' ? 'fa-exclamation-triangle' : 
                 type === 'error' ? 'fa-times-circle' : 'fa-info-circle';
    
    const statusHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert" style="margin: 10px 0;">
            <i class="fas ${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    
    $('.main-content').prepend(statusHtml);
    
    
    setTimeout(() => {
        $('.alert').fadeOut();
    }, 5000);
}


window.dashboardUtils = {
    getDashboardData,
    animateCounter,
    updateProgressBars,
    createChart,
    showStatusMessage
};

function createComparisonChart(clusterData) {
    console.log('📈 إنشاء رسم المقارنة');
    
    if (!clusterData || clusterData.length === 0) return;
    
    const subjects = ['الرياضيات', 'القراءة', 'الكتابة'];
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'];
    
    
    const traces = clusterData.map((cluster, index) => ({
        x: subjects,
        y: [
            75 + (index * -5) + Math.random() * 10, 
            78 + (index * -6) + Math.random() * 8,  
            76 + (index * -5) + Math.random() * 9   
        ],
        type: 'bar',
        name: cluster.name,
        marker: {
            color: colors[index],
            opacity: 0.8,
            line: {
                color: '#FFFFFF',
                width: 2
            }
        },
        text: subjects.map(() => `${cluster.count} طالب`),
        textposition: 'auto',
        hovertemplate: '<b>%{fullData.name}</b><br>' +
                      'المادة: %{x}<br>' +
                      'متوسط الدرجة: %{y:.1f}<br>' +
                      '<extra></extra>'
    }));
    
    const layout = {
        title: {
            text: '📚 مقارنة الأداء بين المجموعات في المواد المختلفة',
            font: { family: 'Cairo', size: 16, color: '#2C2C2C' },
            x: 0.5
        },
        xaxis: {
            title: 'المواد الدراسية',
            font: { family: 'Cairo', size: 12 }
        },
        yaxis: {
            title: 'متوسط الدرجات',
            font: { family: 'Cairo', size: 12 }
        },
        barmode: 'group',
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 60, b: 60, l: 60, r: 60 },
        legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            font: { family: 'Cairo', size: 11 }
        },
        font: { family: 'Cairo' }
    };
    
    
    const comparisonContainer = document.createElement('div');
    comparisonContainer.id = 'comparisonChart';
    comparisonContainer.style.height = '400px';
    comparisonContainer.style.marginTop = '30px';
    comparisonContainer.style.background = '#FFFFFF';
    comparisonContainer.style.borderRadius = '15px';
    comparisonContainer.style.padding = '20px';
    comparisonContainer.style.boxShadow = '0 4px 16px rgba(184, 92, 87, 0.1)';
    
    
    const chartWrapper = document.querySelector('.chart-wrapper');
    if (chartWrapper) {
        chartWrapper.appendChild(comparisonContainer);
        
        Plotly.newPlot('comparisonChart', traces, layout, {
            responsive: true,
            displayModeBar: false
        });
    }
}


function createTrendChart() {
    console.log('📈 إنشاء رسم الاتجاهات');
    
    const months = ['سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر', 'يناير', 'فبراير'];
    
    const traces = [
        {
            x: months,
            y: [250, 245, 255, 260, 265, 270],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'المتفوقون',
            line: { color: '#28a745', width: 3 },
            marker: { size: 8, color: '#28a745' }
        },
        {
            x: months,
            y: [350, 355, 345, 340, 345, 350],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'الجيدون',
            line: { color: '#17a2b8', width: 3 },
            marker: { size: 8, color: '#17a2b8' }
        },
        {
            x: months,
            y: [250, 255, 250, 245, 240, 235],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'المتوسطون',
            line: { color: '#ffc107', width: 3 },
            marker: { size: 8, color: '#ffc107' }
        },
        {
            x: months,
            y: [150, 145, 150, 155, 150, 145],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'يحتاجون دعم',
            line: { color: '#dc3545', width: 3 },
            marker: { size: 8, color: '#dc3545' }
        }
    ];
    
    const layout = {
        title: {
            text: '📊 اتجاهات أداء المجموعات عبر الوقت',
            font: { family: 'Cairo', size: 16, color: '#2C2C2C' },
            x: 0.5
        },
        xaxis: {
            title: 'الشهور',
            font: { family: 'Cairo', size: 12 }
        },
        yaxis: {
            title: 'عدد الطلاب',
            font: { family: 'Cairo', size: 12 }
        },
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 60, b: 60, l: 60, r: 60 },
        legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            font: { family: 'Cairo', size: 11 }
        },
        font: { family: 'Cairo' },
        hovermode: 'x unified'
    };
    
    
    const trendContainer = document.createElement('div');
    trendContainer.id = 'trendChart';
    trendContainer.style.height = '400px';
    trendContainer.style.marginTop = '30px';
    trendContainer.style.background = '#FFFFFF';
    trendContainer.style.borderRadius = '15px';
    trendContainer.style.padding = '20px';
    trendContainer.style.boxShadow = '0 4px 16px rgba(184, 92, 87, 0.1)';
    
    const chartWrapper = document.querySelector('.chart-wrapper');
    if (chartWrapper) {
        chartWrapper.appendChild(trendContainer);
        
        Plotly.newPlot('trendChart', traces, layout, {
            responsive: true,
            displayModeBar: false
        });
    }
}


function createGeographicChart() {
    console.log('🗺️ إنشاء الرسم الجغرافي');
    
    const regions = ['الرياض', 'جدة', 'الدمام', 'مكة', 'المدينة', 'الطائف'];
    const studentCounts = [180, 150, 120, 100, 80, 70];
    
    const trace = {
        x: regions,
        y: studentCounts,
        type: 'bar',
        marker: {
            color: studentCounts,
            colorscale: [
                [0, '#FFE5E5'],
                [0.2, '#FFB3B3'],
                [0.4, '#FF8080'],
                [0.6, '#FF4D4D'],
                [0.8, '#FF1A1A'],
                [1, '#E60000']
            ],
            colorbar: {
                title: 'عدد الطلاب',
                titlefont: { family: 'Cairo' }
            },
            line: { color: '#FFFFFF', width: 2 }
        },
        text: studentCounts.map(count => `${count} طالب`),
        textposition: 'auto',
        textfont: { color: 'white', size: 12, family: 'Cairo', weight: 'bold' }
    };
    
    const layout = {
        title: {
            text: '🏙️ التوزيع الجغرافي للطلاب',
            font: { family: 'Cairo', size: 16, color: '#2C2C2C' },
            x: 0.5
        },
        xaxis: {
            title: 'المناطق',
            font: { family: 'Cairo', size: 12 }
        },
        yaxis: {
            title: 'عدد الطلاب',
            font: { family: 'Cairo', size: 12 }
        },
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 60, b: 60, l: 60, r: 60 },
        font: { family: 'Cairo' }
    };
    
    
    const geoContainer = document.createElement('div');
    geoContainer.id = 'geographicChart';
    geoContainer.style.height = '400px';
    geoContainer.style.marginTop = '30px';
    geoContainer.style.background = '#FFFFFF';
    geoContainer.style.borderRadius = '15px';
    geoContainer.style.padding = '20px';
    geoContainer.style.boxShadow = '0 4px 16px rgba(184, 92, 87, 0.1)';
    
    const chartWrapper = document.querySelector('.chart-wrapper');
    if (chartWrapper) {
        chartWrapper.appendChild(geoContainer);
        
        Plotly.newPlot('geographicChart', [trace], layout, {
            responsive: true,
            displayModeBar: false
        });
    }
}


function createPerformanceIndicators(clusterData) {
    console.log('📊 إنشاء مؤشرات الأداء');
    
    const indicatorsContainer = document.createElement('div');
    indicatorsContainer.className = 'performance-indicators';
    indicatorsContainer.style.cssText = `
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #F5F1ED 0%, #EDE4DB 100%);
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(184, 92, 87, 0.1);
    `;
    
    const indicators = [
        { title: 'معدل النجاح', value: '87%', icon: '🎯', color: '#28a745' },
        { title: 'متوسط الدرجات', value: '78.5', icon: '📈', color: '#17a2b8' },
        { title: 'معدل التحسن', value: '+12%', icon: '⬆️', color: '#ffc107' },
        { title: 'الحضور', value: '94%', icon: '👥', color: '#6f42c1' }
    ];
    
    indicators.forEach(indicator => {
        const indicatorElement = document.createElement('div');
        indicatorElement.style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            border-left: 4px solid ${indicator.color};
        `;
        
        indicatorElement.innerHTML = `
            <div style="font-size: 2rem; margin-bottom: 10px;">${indicator.icon}</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: ${indicator.color}; margin-bottom: 5px;">${indicator.value}</div>
            <div style="font-size: 0.9rem; color: #6B6B6B; font-weight: 600;">${indicator.title}</div>
        `;
        
        indicatorElement.addEventListener('mouseenter', () => {
            indicatorElement.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        indicatorElement.addEventListener('mouseleave', () => {
            indicatorElement.style.transform = 'translateY(0) scale(1)';
        });
        
        indicatorsContainer.appendChild(indicatorElement);
    });
    
    const chartWrapper = document.querySelector('.chart-wrapper');
    if (chartWrapper) {
        chartWrapper.appendChild(indicatorsContainer);
    }
}


window.dashboardUtils = {
    getDashboardData,
    animateCounter,
    updateProgressBars,
    createChart,
    createComparisonChart,
    createTrendChart,
    createGeographicChart,
    createPerformanceIndicators,
    showStatusMessage
};