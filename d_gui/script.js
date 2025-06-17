/**
 * ParaphrAIx - Main Application Logic
 *
 * This script is organized into the following modular sections:
 * 1. STATE & CONSTANTS: Global state, configuration, and constants.
 * 2. INITIALIZATION: Sets up the application on page load.
 * 3. DATA HANDLING: Functions for fetching and processing the JSON data.
 * 4. UI RENDERING: A suite of functions to render different parts of the UI.
 * 5. EVENT HANDLERS: Manages all user interactions (clicks, changes).
 * 6. CHARTING: Functions dedicated to creating/updating charts with Chart.js.
 * 7. SEARCH LOGIC: The specific logic for the "Search" page functionality.
 * 8. UTILITIES: Helper functions used across the application.
 */

// -----------------------------------------------------------------------------
// 1. STATE & CONSTANTS
// -----------------------------------------------------------------------------

const METRICS = [
    "Task Fulfilment / Relevance", "Usefulness & Actionability", "Factual Accuracy & Verifiability",
    "Efficiency / Depth & Completeness", "Reasoning Quality / Transparency", "Tone & Likeability",
    "Adaptation to Context", "Safety & Bias Avoidance", "Structure & Formatting & UX Extras", "Creativity"
];

// File paths are structured for easy extension with new datasets/models.
const DATA_PATHS = {
    instructions: (dataset) => `../a_data/${dataset}/prxed/all.json`,
    scores: (dataset, model) => `../c_assess_inf/output/${dataset}_answer_scores_500/${model}.json`,
    originalInstructions: (dataset) => `../a_data/${dataset}/prxed/all.json`
};

// Global state object to hold current selections and data
let state = {
    currentDataset: 'alpaca',
    currentModel: 'gemma-2-2b-it',
    instructions: [],
    scores: [],
    aggregatedData: {}, // Holds processed averages, stddevs, etc.
    spiderChart: null,
    barChart: null,
    isLoading: true,
};


// -----------------------------------------------------------------------------
// 2. INITIALIZATION
// -----------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', init);

/**
 * Main initialization function, runs after the DOM is fully loaded.
 */
function init() {
    console.log("ParaphrAIx Initializing...");
    setupEventListeners();
    populateStaticElements();
    loadAndProcessData();
}

/**
 * Sets up all the primary event listeners for the application.
 */
function setupEventListeners() {
    // Top control dropdowns
    document.getElementById('dataset-select').addEventListener('change', handleDatasetChange);
    document.getElementById('model-select').addEventListener('change', handleModelChange);

    // Navigation tabs
    document.querySelectorAll('.nav-button').forEach(button => {
        button.addEventListener('click', handleNavClick);
    });

    // Chart update buttons
    document.getElementById('update-spider-chart').addEventListener('click', renderSpiderChart);
    document.getElementById('update-bar-chart').addEventListener('click', renderBarChart);
    
    // Ranking metric selector
    document.getElementById('ranking-metric-select').addEventListener('change', renderRankingList);

    // Search button
    document.getElementById('search-button').addEventListener('click', handleSearch);
}

/**
 * Populates UI elements that don't depend on fetched data (e.g., metric dropdowns).
 */
function populateStaticElements() {
    const rankingSelect = document.getElementById('ranking-metric-select');
    METRICS.forEach((metric, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = metric;
        rankingSelect.appendChild(option);
    });
}


// -----------------------------------------------------------------------------
// 3. DATA HANDLING
// -----------------------------------------------------------------------------

/**
 * Orchestrates fetching data, processing it, and triggering the initial render.
 */
async function loadAndProcessData() {
    setLoading(true);
    try {
        const instructionsPath = DATA_PATHS.instructions(state.currentDataset);
        const scoresPath = DATA_PATHS.scores(state.currentDataset, state.currentModel);

        // Fetch both files in parallel
        const [instructionsRes, scoresRes] = await Promise.all([
            fetch(instructionsPath),
            fetch(scoresPath)
        ]);

        if (!instructionsRes.ok) throw new Error(`Failed to load instructions: ${instructionsRes.statusText}`);
        if (!scoresRes.ok) throw new Error(`Failed to load scores: ${scoresRes.statusText}`);

        state.instructions = await instructionsRes.json();
        state.scores = await scoresRes.json();

        processData();
        renderAll();

    } catch (error) {
        console.error("Error loading data:", error);
        document.getElementById('overview-table-container').innerHTML = `<p style="color:red;">Error: ${error.message}. Please check file paths and ensure the server is running from the correct directory.</p>`;
    } finally {
        setLoading(false);
    }
}

/**
 * Processes the raw loaded data into an aggregated format for easy use.
 * Calculates averages, counts, and standard deviations for each paraphrase style.
 */
function processData() {
    const aggregated = {};
    const paraphraseKeys = new Set();

    // First, find all unique paraphrase types (e.g., 'instruct_apologetic')
    state.scores.forEach(item => {
        Object.keys(item).forEach(key => {
            if (key !== 'prompt_count' && key !== 'prompt_id') {
                paraphraseKeys.add(key);
            }
        });
    });

    // Initialize structures
    paraphraseKeys.forEach(key => {
        aggregated[key] = {
            scores: Array(METRICS.length).fill(0).map(() => []), // Store all scores for std dev
            averages: Array(METRICS.length).fill(0),
            stdDevs: Array(METRICS.length).fill(0),
            overallAverage: 0,
            count: 0,
        };
    });

    // Aggregate scores
    state.scores.forEach(item => {
        paraphraseKeys.forEach(key => {
            if (item[key] && Array.isArray(item[key])) {
                aggregated[key].count++;
                item[key].forEach((score, index) => {
                    aggregated[key].scores[index].push(score);
                });
            }
        });
    });

    // Calculate averages and standard deviations
    paraphraseKeys.forEach(key => {
        if (aggregated[key].count > 0) {
            for (let i = 0; i < METRICS.length; i++) {
                const scoresList = aggregated[key].scores[i];
                if(scoresList.length > 0) {
                    const avg = calculateAverage(scoresList);
                    aggregated[key].averages[i] = avg;
                    aggregated[key].stdDevs[i] = calculateStdDev(scoresList, avg);
                }
            }
            aggregated[key].overallAverage = calculateAverage(aggregated[key].averages);
        }
    });

    state.aggregatedData = aggregated;
    console.log("Processed Data:", state.aggregatedData);
}


// -----------------------------------------------------------------------------
// 4. UI RENDERING
// -----------------------------------------------------------------------------

/**
 * Main render function that calls specific renderers based on the active page.
 */
function renderAll() {
    renderOverviewPage();
    renderRankingPage();
    renderSearchPage();
    
    // Ensure the correct page is visible after a data reload
    const activeNav = document.querySelector('.nav-button.active');
    switchPage(activeNav.id);
}

function renderOverviewPage() {
    renderOverviewTable();
    populateChartSelectors();
    renderSpiderChart();
    renderBarChart();
}

function renderRankingPage() {
    renderRankingList();
}

function renderSearchPage() {
    renderBestPrompts();
}


/**
 * Renders the main data table on the Overview page.
 */
function renderOverviewTable() {
    const container = document.getElementById('overview-table-container');
    let tableHtml = '<table id="overview-table"><thead><tr><th>Paraphrase Style</th>';
    METRICS.forEach(metric => tableHtml += `<th>${metric}</th>`);
    tableHtml += '</tr></thead><tbody>';

    for (const key in state.aggregatedData) {
        if (state.aggregatedData[key].count > 0) {
            tableHtml += `<tr><td><strong>${key}</strong></td>`;
            state.aggregatedData[key].averages.forEach(avg => {
                tableHtml += `<td style="background-color: ${scoreToColor(avg)}">${avg.toFixed(2)}</td>`;
            });
            tableHtml += '</tr>';
        }
    }
    tableHtml += '</tbody></table>';
    container.innerHTML = tableHtml;
}

/**
 * Renders the ranked list and bar chart on the Ranking page.
 */
function renderRankingList() {
    const container = document.getElementById('ranking-container');
    const metricIndex = document.getElementById('ranking-metric-select').value;

    if (!state.aggregatedData || Object.keys(state.aggregatedData).length === 0) return;

    const sortedStyles = Object.entries(state.aggregatedData)
        .filter(([, data]) => data.count > 0)
        .sort(([, a], [, b]) => b.averages[metricIndex] - a.averages[metricIndex]);

    let listHtml = '';
    sortedStyles.forEach(([key, data]) => {
        const score = data.averages[metricIndex];
        const barWidth = (score / 10) * 100;
        listHtml += `
            <div class="ranking-item">
                <div class="ranking-label">${key}</div>
                <div class="ranking-bar-container">
                    <div class="ranking-bar" style="width: ${barWidth}%;">${score.toFixed(2)}</div>
                </div>
            </div>
        `;
    });
    container.innerHTML = listHtml;
}

/**
 * Renders the list of best-performing prompts on the Search page.
 */
function renderBestPrompts() {
    const container = document.getElementById('best-prompts-container');
    if (state.instructions.length === 0 || state.scores.length === 0) {
        container.innerHTML = "<p>Data not available to show best prompts.</p>";
        return;
    }

    const promptScores = state.scores.map(scoreItem => {
        const avgScore = calculateAverage(Object.values(scoreItem)
            .flat() // Flatten in case of nested arrays (though scores are just arrays)
            .filter(v => typeof v === 'number')
        );
        return { id: scoreItem.prompt_id, avgScore };
    }).sort((a, b) => b.avgScore - a.avgScore).slice(0, 5); // Get top 5

    // Find original prompt texts for the top 5
    const topPrompts = promptScores.map(ps => {
        const instructionItem = state.instructions.find(i => i.prompt_id === ps.id);
        return {
            text: instructionItem ? instructionItem.instruction_original : "Prompt not found",
            score: ps.avgScore
        };
    });

    // Simple common word highlighting
    const allWords = topPrompts.map(p => p.text.toLowerCase().split(/\s+/)).flat();
    const wordCounts = allWords.reduce((acc, word) => {
        const cleanWord = word.replace(/[.,?]/g, '');
        if (cleanWord.length > 3 && isNaN(cleanWord)) { // Ignore short words and numbers
           acc[cleanWord] = (acc[cleanWord] || 0) + 1;
        }
        return acc;
    }, {});
    const commonWords = new Set(Object.entries(wordCounts).filter(([,count])=>count > 1).map(([word,])=>word));

    let html = '';
    topPrompts.forEach(prompt => {
        let highlightedText = prompt.text.split(' ').map(word => {
            const cleanWord = word.toLowerCase().replace(/[.,?]/g, '');
            return commonWords.has(cleanWord) ? `<span class="highlight">${word}</span>` : word;
        }).join(' ');

        html += `
            <div class="best-prompt">
                <p class="prompt-text">${highlightedText}</p>
                <p class="prompt-score">Average Score: ${prompt.score.toFixed(2)}</p>
            </div>
        `;
    });
    container.innerHTML = html;
}

/**
 * Toggles the loading indicator visibility.
 */
function setLoading(isLoading) {
    state.isLoading = isLoading;
    document.getElementById('loader').style.display = isLoading ? 'block' : 'none';
    document.querySelector('main').style.display = isLoading ? 'none' : 'block';
}


// -----------------------------------------------------------------------------
// 5. EVENT HANDLERS
// -----------------------------------------------------------------------------

function handleDatasetChange(e) {
    state.currentDataset = e.target.value;
    console.log(`Dataset changed to: ${state.currentDataset}`);
    loadAndProcessData();
}

function handleModelChange(e) {
    state.currentModel = e.target.value;
    console.log(`Model changed to: ${state.currentModel}`);
    loadAndProcessData();
}

function handleNavClick(e) {
    const targetId = e.currentTarget.id;
    switchPage(targetId);
}

function handleSearch() {
    const query = document.getElementById('search-input').value;
    analyzeAndDisplaySearch(query);
}

/**
 * Handles switching between the main pages/tabs.
 */
function switchPage(targetNavId) {
    document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(targetNavId).classList.add('active');

    const pageId = targetNavId.replace('nav-', '') + '-page';
    document.querySelectorAll('.page').forEach(page => {
        page.style.display = page.id === pageId ? 'block' : 'none';
    });
}


// -----------------------------------------------------------------------------
// 6. CHARTING
// -----------------------------------------------------------------------------

/**
 * Populates the multi-select boxes for choosing what to display on charts.
 */
function populateChartSelectors() {
    const spiderSelect = document.getElementById('spider-select');
    const barSelect = document.getElementById('bar-select');
    spiderSelect.innerHTML = '';
    barSelect.innerHTML = '';

    const sortedStyles = Object.keys(state.aggregatedData).sort();

    sortedStyles.forEach(key => {
        if (state.aggregatedData[key].count > 0) {
            const option = new Option(key, key);
            spiderSelect.add(option.cloneNode(true));
            barSelect.add(option);
        }
    });

    // Pre-select some defaults for initial view
    const defaultSelections = ['instruction_original', 'instruct_apologetic', 'instruct_direct'].filter(s => sortedStyles.includes(s));
    for (const option of spiderSelect.options) {
        if (defaultSelections.includes(option.value)) option.selected = true;
    }
     for (const option of barSelect.options) {
        if (defaultSelections.includes(option.value)) option.selected = true;
    }
}

/**
 * Renders the spider (radar) chart on the Overview page.
 */
function renderSpiderChart() {
    const ctx = document.getElementById('spider-chart').getContext('2d');
    const selectedOptions = Array.from(document.getElementById('spider-select').selectedOptions).map(opt => opt.value);
    
    const datasets = selectedOptions.map((key, index) => {
        const color = `hsl(${(index * 100) % 360}, 70%, 50%)`;
        return {
            label: key,
            data: state.aggregatedData[key].averages,
            borderColor: color,
            backgroundColor: `${color}33`, // semi-transparent fill
            pointBackgroundColor: color,
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: color
        };
    });

    if (state.spiderChart) state.spiderChart.destroy();
    
    state.spiderChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: METRICS.map(m => m.split(' ')[0]), // Use shorter labels for chart
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    angleLines: { display: true },
                    suggestedMin: 0,
                    suggestedMax: 10,
                    pointLabels: {
                        font: { size: 10 }
                    }
                }
            },
            plugins: {
                legend: { position: 'top' }
            }
        }
    });
}

/**
 * Renders the bar chart on the Overview page.
 * This chart shows the overall average score for selected paraphrase styles.
 */
function renderBarChart() {
    const ctx = document.getElementById('bar-chart').getContext('2d');
    const selectedOptions = Array.from(document.getElementById('bar-select').selectedOptions).map(opt => opt.value);

    const labels = selectedOptions;
    const dataPoints = selectedOptions.map(key => state.aggregatedData[key].overallAverage);
    const backgroundColors = dataPoints.map(score => scoreToColor(score, 0.8)); // 80% opacity
    const borderColors = dataPoints.map(score => scoreToColor(score, 1.0)); // Full opacity

    if (state.barChart) state.barChart.destroy();

    state.barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Overall Average Score',
                data: dataPoints,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y', // Horizontal bars
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 10
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}


// -----------------------------------------------------------------------------
// 7. SEARCH LOGIC
// -----------------------------------------------------------------------------

const SEARCH_PATTERNS = {
    // Paraphrase Styles
    instruct_apologetic: /\b(sorry|bother|please|kindly|if you could|trouble)\b/gi,
    instruct_archaic: /\b(hark|thee|thou|beseech|pray|whence|henceforth)\b/gi,
    instruct_colloquial: /\b(gonna|wanna|kinda|sorta|like|you know)\b/gi,
    instruct_direct: /^\s*(give|list|explain|what is|tell me|generate|write)/gi,
    instruct_formal: /\b(furthermore|consequently|regarding|it is imperative|would be appreciated)\b/gi,
    // Datasets (Topics)
    topic_gsm8k: /\b(calculate|solve|how many|what is the total|number|equation|math)\b/gi,
    topic_mmlu: /\b(philosophy|law|history|computer science|economics|biology|chemistry)\b/gi,
};

function analyzeAndDisplaySearch(query) {
    const resultsContainer = document.getElementById('search-results-container');
    if (!query.trim()) {
        resultsContainer.innerHTML = '<p>Please enter a prompt to analyze.</p>';
        return;
    }

    // Match paraphrase style
    let bestStyleMatch = 'instruction_original'; // Default
    let maxStyleMatches = 0;
    for (const [style, pattern] of Object.entries(SEARCH_PATTERNS)) {
        if (!style.startsWith('topic_')) {
            const matches = (query.match(pattern) || []).length;
            if (matches > maxStyleMatches) {
                maxStyleMatches = matches;
                bestStyleMatch = style;
            }
        }
    }

    // Match topic
    let bestTopicMatch = 'alpaca'; // Default
    if (SEARCH_PATTERNS.topic_gsm8k.test(query)) bestTopicMatch = 'gsm8k';
    if (SEARCH_PATTERNS.topic_mmlu.test(query)) bestTopicMatch = 'mmlu';

    // Get score for matched style IF the topic matches the current dataset
    let resultHTML = `<p><strong>Analysis Results:</strong></p>`;
    resultHTML += `<p>Detected Prompt Style: <strong>${bestStyleMatch}</strong></p>`;
    resultHTML += `<p>Detected Topic: <strong>${bestTopicMatch}</strong> (Current dataset: ${state.currentDataset})</p>`;
    
    if (bestTopicMatch !== state.currentDataset) {
        resultHTML += `<p style="color:orange;">Warning: Prompt topic may not match the currently loaded '${state.currentDataset}' dataset. Performance prediction might be inaccurate. Please switch datasets for a better prediction.</p>`;
    }
    
    const predictedPerf = state.aggregatedData[bestStyleMatch];
    if(predictedPerf) {
         resultHTML += `<p>Predicted Average Score: <strong style="font-size: 1.2em;">${predictedPerf.overallAverage.toFixed(2)} / 10</strong></p>`;
    } else {
         resultHTML += `<p>Could not retrieve performance data for the detected style.</p>`;
    }

    // Find best performing style for the detected topic
    const bestStyleForTopic = findBestPerformingStyle();
    const recommendedWords = getKeywordsForStyle(bestStyleForTopic);

    resultHTML += `<hr><p><strong>Recommendation:</strong></p>`;
    resultHTML += `<p>For the '${state.currentDataset}' dataset, the best performing style is <strong>'${bestStyleForTopic}'</strong>.</p>`;
    if (recommendedWords) {
        resultHTML += `<p>Consider using words like: <em>${recommendedWords}</em></p>`;
    }

    resultsContainer.innerHTML = resultHTML;
}

function findBestPerformingStyle() {
    return Object.entries(state.aggregatedData)
        .filter(([, data]) => data.count > 0)
        .reduce((best, current) => {
            return current[1].overallAverage > best[1].overallAverage ? current : best;
        })[0];
}

function getKeywordsForStyle(style) {
    const pattern = SEARCH_PATTERNS[style];
    if (!pattern) return "N/A";
    // Extract keywords from regex
    return pattern.source.replace(/\\b/g, '').replace(/[()|]/g, ' ').replace(/\s+/g, ' ').trim().split(' ').join(', ');
}


// -----------------------------------------------------------------------------
// 8. UTILITIES
// -----------------------------------------------------------------------------

function calculateAverage(arr) {
    if (!arr || arr.length === 0) return 0;
    const sum = arr.reduce((acc, val) => acc + val, 0);
    return sum / arr.length;
}

function calculateStdDev(arr, mean) {
    if (!arr || arr.length < 2) return 0;
    const avg = mean !== undefined ? mean : calculateAverage(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    const avgSquareDiff = calculateAverage(squareDiffs);
    return Math.sqrt(avgSquareDiff);
}

/**
 * Converts a score (0-10) to a color from white to green.
 * @param {number} score - The score from 0 to 10.
 * @param {number} opacity - The alpha/opacity value from 0 to 1.
 * @returns {string} An hsla color string.
 */
function scoreToColor(score, opacity = 0.4) {
    const normalizedScore = Math.max(0, Math.min(10, score)) / 10;
    // Hue for green is around 120. We can map score to lightness.
    // Low score -> high lightness (white-ish green)
    // High score -> lower lightness (darker green)
    const lightness = 95 - (normalizedScore * 45); // from 95% (very light) to 50% (full color)
    const saturation = 80;
    return `hsla(120, ${saturation}%, ${lightness}%, ${opacity})`;
}