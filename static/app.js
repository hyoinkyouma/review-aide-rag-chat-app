// ===== STATE =====
let currentTab = 'chat';
let isSending = false;
let polling = false;
let dlPolling = false;
let webSearchEnabled = false;

// ===== DOM REFS =====
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const tabChat = $('#tab-chat');
const tabSettings = $('#tab-settings');
const tabBtns = $$('.tab-btn');
const messagesEl = $('#messages');
const chatForm = $('#chat-form');
const chatInput = $('#chat-input');
const sendBtn = $('#send-btn');
const noModelBanner = $('#no-model-banner');
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');
const fileList = $('#file-list');
const ingestBtn = $('#ingest-btn');
const clearBtn = $('#clear-btn');
const progressArea = $('#progress-area');
const progressBar = $('#progress-bar');
const progressMsg = $('#progress-message');
const progressPct = $('#progress-percent');
const webSearchToggle = $('#web-search-toggle');
const modelList = $('#model-list');
const dlProgressArea = $('#dl-progress-area');
const dlProgressBar = $('#dl-progress-bar');
const dlProgressMsg = $('#dl-progress-message');
const dlProgressPct = $('#dl-progress-pct');

// ===== TAB SWITCHING =====
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  currentTab = tab;
  tabBtns.forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  tabChat.classList.toggle('hidden', tab !== 'chat');
  tabChat.classList.toggle('flex', tab === 'chat');
  tabSettings.classList.toggle('hidden', tab !== 'settings');
  tabSettings.classList.toggle('flex', tab === 'settings');
  if (tab === 'settings') { refreshFileList(); refreshModelList(); }
}

// ===== MODEL LIST =====
async function refreshModelList() {
  try {
    const resp = await fetch('/v1/models');
    const data = await resp.json();
    renderModelList(data.models, data.current_model);
  } catch (e) {
    console.error('Failed to fetch models:', e);
  }
}

function renderModelList(models, currentId) {
  modelList.innerHTML = models.map(m => {
    let badge = '';
    let btn = '';
    if (m.active) {
      badge = '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">Active</span>';
    } else if (m.downloaded) {
      btn = `<button onclick="selectModel('${m.id}')" class="px-3 py-1 text-xs font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700">Activate</button>`;
      badge = '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">Downloaded</span>';
    } else {
      btn = `<button onclick="downloadModel('${m.id}')" class="px-3 py-1 text-xs font-medium rounded-md bg-gray-200 text-gray-700 hover:bg-gray-300">Download</button>`;
      badge = '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-400">Not downloaded</span>';
    }
    const border = m.active ? 'border-blue-500 active' : 'border-gray-200';
    return `<div class="model-card rounded-xl border ${border} p-4 flex items-start justify-between gap-4">
      <div class="min-w-0">
        <div class="flex items-center gap-2 mb-1">
          <span class="font-medium text-sm text-gray-800">${m.name}</span>
          ${badge}
        </div>
        <p class="text-xs text-gray-500">${m.description}</p>
        <p class="text-xs text-gray-400 mt-1">${m.size_human} &middot; ${m.repo_id}</p>
      </div>
      <div class="shrink-0">${btn}</div>
    </div>`;
  }).join('');
}

async function downloadModel(key) {
  try {
    const resp = await fetch(`/v1/models/download/${key}`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'started') {
      dlProgressArea.classList.remove('hidden');
      dlProgressBar.style.width = '0%';
      dlProgressPct.textContent = '0%';
      dlProgressMsg.textContent = 'Starting download...';
      if (!dlPolling) startDlPolling();
    } else if (data.status === 'already_downloaded') {
      refreshModelList();
    }
  } catch (e) {
    alert('Failed to start download: ' + e.message);
  }
}

function startDlPolling() {
  dlPolling = true;
  const interval = setInterval(async () => {
    try {
      const resp = await fetch('/v1/models/download/progress');
      const prog = await resp.json();
      dlProgressBar.style.width = prog.progress + '%';
      dlProgressPct.textContent = prog.progress + '%';
      dlProgressMsg.textContent = prog.message || 'Downloading...';

      if (prog.status === 'completed') {
        clearInterval(interval);
        dlPolling = false;
        dlProgressMsg.textContent = 'Download complete!';
        setTimeout(() => { dlProgressArea.classList.add('hidden'); }, 3000);
        refreshModelList();
      } else if (prog.status === 'error') {
        clearInterval(interval);
        dlPolling = false;
        dlProgressBar.style.width = '0%';
        dlProgressMsg.textContent = 'Download failed. Check the server logs.';
      }
    } catch (e) {
      console.error('DL poll error:', e);
    }
  }, 1000);
}

async function selectModel(key) {
  try {
    const resp = await fetch(`/v1/models/select/${key}`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      refreshModelList();
      checkHealth();
      addMessage('assistant', 'Switched to model. You can now ask questions.');
    }
  } catch (e) {
    alert('Failed to select model: ' + e.message);
  }
}

// ===== HEALTH / FIRST-START =====
async function checkHealth() {
  try {
    const resp = await fetch('/health');
    const data = await resp.json();
    if (data.model_loaded) {
      noModelBanner.classList.add('hidden');
      chatInput.disabled = false;
      sendBtn.disabled = false;
    } else {
      noModelBanner.classList.remove('hidden');
      chatInput.disabled = true;
      sendBtn.disabled = true;
    }
  } catch (e) {
    console.error('Health check failed:', e);
  }
}

// ===== MARKDOWN RENDERER =====
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  const paras = html.split(/\n\n+/);
  return paras.map(p => {
    p = p.trim();
    if (!p) return '';
    p = p.replace(/\n/g, '<br>');
    return `<p>${p}</p>`;
  }).join('');
}

// ===== CHAT =====
function addMessage(role, content, citations) {
  const div = document.createElement('div');
  div.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
  const inner = document.createElement('div');
  inner.className = `max-w-[80%] md:max-w-[70%] px-4 py-2.5 msg-content ${role === 'user' ? 'msg-user' : 'msg-assistant'}`;
  inner.innerHTML = renderMarkdown(content);
  div.appendChild(inner);

  if (role === 'assistant' && citations && citations.length > 0) {
    const citeWrapper = document.createElement('div');
    citeWrapper.className = 'ml-2 mt-1 space-y-1';
    citations.forEach((c) => {
      const cite = document.createElement('div');
      cite.className = 'text-xs';
      const toggle = document.createElement('button');
      toggle.className = 'citation-toggle text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1';
      toggle.innerHTML = `<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 12h6m-6 4h6m2-10H7a2 2 0 00-2 2v14l4-2 4 2 4-2 4 2V6a2 2 0 00-2-2z"/></svg> ${c.source}${c.page != null ? ' (p.' + c.page + ')' : ''}`;
      const body = document.createElement('div');
      body.className = 'citation-body pl-4 text-gray-500';
      body.textContent = c.content;
      toggle.addEventListener('click', () => body.classList.toggle('open'));
      cite.appendChild(toggle);
      cite.appendChild(body);
      citeWrapper.appendChild(cite);
    });
    div.appendChild(citeWrapper);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTypingIndicator() {
  const div = document.createElement('div');
  div.id = 'typing-indicator';
  div.className = 'flex justify-start';
  div.innerHTML = `<div class="bg-gray-100 rounded-2xl px-4 py-3 typing-indicator flex gap-1"><span></span><span></span><span></span></div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

webSearchToggle.addEventListener('click', () => {
  webSearchEnabled = !webSearchEnabled;
  webSearchToggle.classList.toggle('active', webSearchEnabled);
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text || isSending) return;

  chatInput.value = '';
  sendBtn.disabled = true;
  isSending = true;

  const msgHtml = webSearchEnabled
    ? '<span class="inline-flex items-center gap-1"><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v6l4 2"/></svg> ' + text + '</span>'
    : text;
  addMessage('user', msgHtml);
  addTypingIndicator();

  try {
    const body = {
      model: 'default',
      messages: [{ role: 'user', content: text }],
      temperature: 0.1,
      max_tokens: 512,
    };
    if (webSearchEnabled) body.web_search = true;
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const errData = await resp.json();
      throw new Error(errData.detail || resp.statusText);
    }
    const data = await resp.json();
    removeTypingIndicator();
    const answer = data.choices?.[0]?.message?.content || '(empty response)';
    addMessage('assistant', answer, data.citations);
  } catch (err) {
    removeTypingIndicator();
    addMessage('assistant', 'Error: ' + err.message);
  } finally {
    isSending = false;
    sendBtn.disabled = false;
  }
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

// ===== FILE UPLOAD =====
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
  fileInput.value = '';
});

async function uploadFiles(files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  try {
    const resp = await fetch('/v1/files/upload', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.status === 'ok') refreshFileList();
  } catch (err) {
    alert('Upload failed: ' + err.message);
  }
}

async function refreshFileList() {
  try {
    const resp = await fetch('/v1/files');
    const data = await resp.json();
    const uf = data.files || [];
    renderFileList(uf);
  } catch (err) {
    console.error('Failed to list files:', err);
  }
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function renderFileList(files) {
  if (!files.length) {
    fileList.innerHTML = '<p class="text-sm text-gray-400 text-center py-4">No files uploaded yet</p>';
    ingestBtn.disabled = true;
    return;
  }
  ingestBtn.disabled = false;
  fileList.innerHTML = files.map((f) =>
    `<div class="file-item flex items-center justify-between px-3 py-2 rounded-lg border border-gray-200">
      <div class="flex items-center gap-2 min-w-0">
        <svg class="w-5 h-5 text-gray-400 shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M19.5 14.25v-6.3a1.5 1.5 0 00-.44-1.06L14.6 3.44a1.5 1.5 0 00-1.06-.44H6.75A2.25 2.25 0 004.5 5.25v13.5A2.25 2.25 0 006.75 21h6.75"/><path d="M14.25 3.75v4.5a.75.75 0 00.75.75h4.5"/></svg>
        <span class="text-sm text-gray-700 truncate">${f.name}</span>
        <span class="text-xs text-gray-400 shrink-0">${formatSize(f.size)}</span>
      </div>
      <button onclick="deleteFile('${f.name}')" class="text-gray-400 hover:text-red-500 p-1 shrink-0">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>`
  ).join('');
}

async function deleteFile(name) {
  try {
    await fetch('/v1/files/' + encodeURIComponent(name), { method: 'DELETE' });
    refreshFileList();
  } catch (err) {
    console.error('Delete failed:', err);
  }
}

clearBtn.addEventListener('click', async () => {
  try {
    await fetch('/v1/files/clear', { method: 'POST' });
    refreshFileList();
  } catch (err) {
    console.error('Clear failed:', err);
  }
});

// ===== INGESTION =====
ingestBtn.addEventListener('click', async () => {
  if (ingestBtn.disabled) return;
  try {
    const resp = await fetch('/v1/ingest', { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'started') {
      progressArea.classList.remove('hidden');
      progressBar.style.width = '0%';
      progressPct.textContent = '0%';
      progressMsg.textContent = 'Starting ingestion...';
      ingestBtn.disabled = true;
      if (!polling) startPolling();
    }
  } catch (err) {
    alert('Failed to start ingestion: ' + err.message);
  }
});

function startPolling() {
  polling = true;
  const interval = setInterval(async () => {
    try {
      const resp = await fetch('/v1/ingest/progress');
      const prog = await resp.json();
      const pct = prog.total > 0 ? Math.round((prog.current / prog.total) * 100) : 0;
      progressBar.style.width = pct + '%';
      progressPct.textContent = pct + '%';
      progressMsg.textContent = prog.message || 'Processing...';
      if (prog.status === 'completed' || prog.status === 'error' || prog.status === 'idle') {
        clearInterval(interval);
        polling = false;
        ingestBtn.disabled = false;
        if (prog.status === 'completed') {
          progressMsg.textContent = 'Done! ' + (prog.message || '');
          refreshFileList();
          setTimeout(() => { progressArea.classList.add('hidden'); }, 3000);
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
    }
  }, 1000);
}

// ===== DARK MODE =====
const darkToggle = $('#dark-toggle');
const html = document.documentElement;

function setTheme(dark) {
  if (dark) {
    html.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  } else {
    html.classList.remove('dark');
    localStorage.setItem('theme', 'light');
  }
}

if (localStorage.getItem('theme') === 'dark' ||
    (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
  html.classList.add('dark');
}

darkToggle.addEventListener('click', () => {
  setTheme(!html.classList.contains('dark'));
});

// ===== INIT =====
async function init() {
  await checkHealth();
  addMessage('assistant', 'Hello! I can answer questions about your documents. Upload files in Settings to expand my knowledge. If no model is loaded, go to Settings to download and activate one.');
}

init();
