let predictionsData = [];
let shotsData = [];
let metricsData = null;
let edaData = null;

const canvas = document.getElementById('pitch');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

function openTab(tabId) {
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    
    // Find the button that called this and add active class to it
    const btns = document.querySelectorAll('.tab-btn');
    for(let btn of btns) {
        if(btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    }
}

async function init() {
    try {
        const [predRes, shotsRes, metricsRes, edaRes] = await Promise.all([
            fetch('predictions.json').then(r => r.ok ? r.json() : []),
            fetch('shots_with_xg.json').then(r => r.ok ? r.json() : []),
            fetch('metrics.json').then(r => r.ok ? r.json() : null),
            fetch('eda_insights.json').then(r => r.ok ? r.json() : null)
        ]);
        
        predictionsData = await predRes;
        shotsData = await shotsRes;
        metricsData = await metricsRes;
        edaData = await edaRes;
        
        document.getElementById('loading-indicator').style.display = 'none';

        setupSelectors();
        renderEDACharts();
        renderPerformance();
        initShotMap();
        setupEventListeners();
        
        // Initial animation
        document.querySelectorAll('.card').forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('loading-indicator').innerText = 'Error cargando datos';
    }
}

function setupSelectors() {
    const hSelect = document.getElementById('home-team-select');
    const aSelect = document.getElementById('away-team-select');
    const tSelect = document.getElementById('team-filter');
    
    // Get unique teams from predictionsData
    const teams = [...new Set(predictionsData.map(p => p.home))].sort();
    
    teams.forEach(team => {
        if(hSelect) hSelect.add(new Option(team, team));
        if(aSelect) aSelect.add(new Option(team, team));
        if(tSelect) tSelect.add(new Option(team, team));
    });
    
    if(teams.length > 1) {
        if(hSelect) hSelect.value = teams[0];
        if(aSelect) aSelect.value = teams[1];
    }
}

function renderPerformance() {
    if (!metricsData) return;
    
    document.getElementById('perf-auc').innerText = metricsData.xg_model.auc.toFixed(3);
    document.getElementById('perf-f1').innerText = metricsData.xg_model.f1.toFixed(3);
    const acc = metricsData.match_predictor.accuracy_cv;
    document.getElementById('perf-acc').innerText = (acc * 100).toFixed(1) + '%';
    
    // Confusion Matrix
    const cm = metricsData.match_predictor.confusion_matrix;
    const flat = cm.flat();
    const maxVal = Math.max(...flat);
    
    for (let i=0; i<3; i++) {
        for (let j=0; j<3; j++) {
            const val = cm[i][j];
            const cell = document.getElementById(`cm-${i}${j}`);
            if (cell) {
                cell.innerText = val;
                cell.style.backgroundColor = `rgba(99,102,241,${(val/maxVal)*0.8})`;
            }
        }
    }
    
    // Benchmarks
    const bar = document.getElementById('perf-bar');
    const txt = document.getElementById('perf-bar-txt');
    if (bar && txt) {
        setTimeout(() => {
            bar.style.height = `${acc * 100}%`;
            txt.innerText = `${(acc * 100).toFixed(1)}%`;
            if (acc > 0.498) {
                bar.style.background = '#10b981';
                txt.style.color = '#10b981';
            } else {
                bar.style.background = '#ef4444';
                txt.style.color = '#ef4444';
            }
        }, 500);
    }
}

function renderEDACharts() {
    if (!edaData) return;
    
    Chart.defaults.color = '#cbd5e1';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    // 1. Conversion by distance
    const distCtx = document.getElementById('edaDistChart');
    if (distCtx && edaData.conversion_by_distance) {
        const labels = Object.keys(edaData.conversion_by_distance);
        const data = Object.values(edaData.conversion_by_distance).map(v => v * 100);
        new Chart(distCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Tasa de Conversión (%)',
                    data: data,
                    backgroundColor: '#8b5cf6'
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: 'Conversión por Distancia al Arco', color: 'white', font: {size: 16} },
                    tooltip: { callbacks: { label: c => c.raw.toFixed(1) + '%' } }
                }
            }
        });
    }

    // 2. H/D/A
    const hdaCtx = document.getElementById('edaHDAChart');
    if (hdaCtx && edaData.results_distribution) {
        new Chart(hdaCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(edaData.results_distribution),
                datasets: [{
                    data: Object.values(edaData.results_distribution).map(v => v * 100),
                    backgroundColor: ['#10b981', '#64748b', '#ef4444']
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: { animateRotate: true, duration: 1200 },
                plugins: {
                    title: { display: true, text: 'Distribución de Resultados', color: 'white', font: {size: 16} }
                }
            }
        });
    }

    // 3. Top Players
    const topCtx = document.getElementById('edaTopChart');
    if (topCtx && edaData.top_players_xg) {
        const players = edaData.top_players_xg;
        new Chart(topCtx, {
            type: 'bar',
            data: {
                labels: players.map(p => p.web_name || p.player_name),
                datasets: [
                    { label: 'xG', data: players.map(p => p.expected_goals), backgroundColor: '#8b5cf6' },
                    { label: 'Goles Reales', data: players.map(p => p.goals_scored || p.goals || 0), backgroundColor: '#f59e0b' }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: 'Top 10 Jugadores: xG vs Goles Reales', color: 'white', font: {size: 16} }
                }
            }
        });
    }
}

// Shot Map
function xgToColor(xg) {
    const value = Math.max(0, Math.min(1, xg));
    const r = Math.round(239 - (239 - 34) * value);
    const g = Math.round(68 + (197 - 68) * value);
    const b = Math.round(68 - (68 - 94) * value);
    return `rgb(${r}, ${g}, ${b})`;
}

function drawPitch() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth = 1.5;
    
    // Contorno
    ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
    // Linea central
    ctx.beginPath(); ctx.moveTo(canvas.width / 2, 10); ctx.lineTo(canvas.width / 2, canvas.height - 10); ctx.stroke();
    // Circulo central
    ctx.beginPath(); ctx.arc(canvas.width / 2, canvas.height / 2, canvas.width * 0.09, 0, 2 * Math.PI); ctx.stroke();
    
    // Área grande izquierda (x=0 to 16.5, y=21 to 79) -> El arco está a la derecha en Opta (100)
    // Entonces para arco rival (donde se ataca), dibujamos el área derecha
    ctx.strokeRect(canvas.width - 10 - canvas.width * 0.165, canvas.height * 0.21, canvas.width * 0.165, canvas.height * 0.58);
    // Área pequeña derecha (x=100-5.5 to 100, y=36 to 64)
    ctx.strokeRect(canvas.width - 10 - canvas.width * 0.055, canvas.height * 0.36, canvas.width * 0.055, canvas.height * 0.28);
}

function drawShots(filterTeam = 'all', showGoalsOnly = false) {
    drawPitch();
    
    shotsData.forEach(shot => {
        if (filterTeam !== 'all' && shot.team !== filterTeam) return;
        if (showGoalsOnly && !shot.is_goal) return;
        
        const cx = (shot.x / 100) * canvas.width;
        const cy = (shot.y / 100) * canvas.height;
        const r = Math.max(4, Math.min(14, shot.xg_predicted * 40));
        const color = xgToColor(shot.xg_predicted);
        
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.fillStyle = color; ctx.fill();
        if (shot.is_goal) {
            ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
        }
    });
}

function initShotMap() {
    if (!canvas || shotsData.length === 0) return;
    drawShots();
    
    document.getElementById('team-filter').addEventListener('change', (e) => {
        drawShots(e.target.value, document.getElementById('show-goals-only').checked);
    });
    
    document.getElementById('show-goals-only').addEventListener('change', (e) => {
        drawShots(document.getElementById('team-filter').value, e.target.checked);
    });
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        let closestShot = null;
        let minDist = Infinity;
        
        const filterTeam = document.getElementById('team-filter').value;
        const showGoalsOnly = document.getElementById('show-goals-only').checked;
        
        shotsData.forEach(shot => {
            if (filterTeam !== 'all' && shot.team !== filterTeam) return;
            if (showGoalsOnly && !shot.is_goal) return;
            
            const cx = (shot.x / 100) * canvas.width;
            const cy = (shot.y / 100) * canvas.height;
            const dist = Math.hypot(cx - mouseX, cy - mouseY);
            
            if (dist < 15 && dist < minDist) {
                minDist = dist;
                closestShot = shot;
            }
        });
        
        if (closestShot) {
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY + 15) + 'px';
            tooltip.innerHTML = `
                <strong>${closestShot.player_name || closestShot.web_name || 'Desconocido'}</strong><br>
                <span style="color:#94a3b8">${closestShot.team}</span><br>
                xG: <span style="color:#4ade80">${closestShot.xg_predicted.toFixed(3)}</span><br>
                Distancia: ${closestShot.distance_to_goal.toFixed(1)}m<br>
                ${closestShot.is_goal ? '<span style="color:#f59e0b; font-weight:bold;">[GOL]</span>' : ''}
            `;
            canvas.style.cursor = 'pointer';
        } else {
            tooltip.style.display = 'none';
            canvas.style.cursor = 'crosshair';
        }
    });
    
    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

function setupEventListeners() {
    const btn = document.getElementById('simulate-btn');
    if (btn) btn.addEventListener('click', simulateMatch);
}

function simulateMatch() {
    const hTeam = document.getElementById('home-team-select').value;
    const aTeam = document.getElementById('away-team-select').value;
    
    if (hTeam === aTeam) {
        alert('Selecciona dos equipos diferentes');
        return;
    }
    
    const pred = predictionsData.find(p => p.home === hTeam && p.away === aTeam);
    
    if (!pred) {
        alert('No hay predicciones para este enfrentamiento.');
        return;
    }
    
    // UI Updates with animation
    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

    animateNumber('res-home-goals', Math.max(0, pred.expected_goals/2), 1);
    animateNumber('res-away-goals', Math.max(0, pred.expected_goals/2), 1);
    
    animateNumber('prob-h-text', pred.prob_H * 100, 1, '%');
    animateNumber('prob-d-text', pred.prob_D * 100, 1, '%');
    animateNumber('prob-a-text', pred.prob_A * 100, 1, '%');

    document.getElementById('prob-h-bar').style.width = `${pred.prob_H * 100}%`;
    document.getElementById('prob-d-bar').style.width = `${pred.prob_D * 100}%`;
    document.getElementById('prob-a-bar').style.width = `${pred.prob_A * 100}%`;
    
    document.getElementById('res-home-name').innerText = hTeam.toUpperCase();
    document.getElementById('res-away-name').innerText = aTeam.toUpperCase();
}

function animateNumber(id, endValue, decimals = 0, suffix = '') {
    const obj = document.getElementById(id);
    if (!obj) return;
    let startValue = parseFloat(obj.innerText) || 0;
    const duration = 1000;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = progress * (endValue - startValue) + startValue;
        obj.innerText = current.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

init();
