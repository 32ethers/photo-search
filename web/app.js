const API = '';

// ========== State ==========
let currentTab = 'search';
let currentClusterId = null;
let clustersCache = [];
let mergeMode = false;
let selectedForMerge = new Set();
// 搜索分页状态
let lastSearchParams = null;
let searchOffset = 0;
const PAGE_SIZE = 30;

// ========== Tab Switching ==========
function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');

    // 人物页隐藏内容搜索行，搜索页显示
    document.getElementById('content-row').style.display = tab === 'search' ? '' : 'none';

    if (tab === 'search') {
        show('page-search');
        hide('page-people');
    } else {
        hide('page-search');
        show('page-people');
        if (mergeMode) toggleMergeMode();
        loadPeople();
    }
}

// ========== Search ==========
function doSearch(append) {
    // 人物详情页：用搜索栏筛选当前人物的照片
    if (currentTab === 'people' && currentClusterId) {
        const params = new URLSearchParams();
        const df = document.getElementById('c-date-from').value;
        const dt = document.getElementById('c-date-to').value;
        const loc = document.getElementById('c-location').value.trim();
        const dev = document.getElementById('c-device').value.trim();
        if (df) params.set('date_from', df);
        if (dt) params.set('date_to', dt);
        if (loc) params.set('location', loc);
        if (dev) params.set('device', dev);
        const qs = params.toString();
        fetch(`${API}/api/faces/clusters/${currentClusterId}/photos${qs ? '?' + qs : ''}`)
            .then(r => r.json())
            .then(data => {
                const count = data.photos?.length || 0;
                document.getElementById('detail-count').textContent = `${count} 张照片`;
                renderPhotoGrid(data, 'detail-grid', false, false);
            });
        return;
    }

    // 搜索页：正常搜索
    const text = val('c-text');
    const date_from = val('c-date-from');
    const date_to = val('c-date-to');
    const location = val('c-location');
    const device = val('c-device');

    if (!text && !date_from && !date_to && !location && !device) return;

    if (!append) {
        searchOffset = 0;
        lastSearchParams = { query: text || '', date_from, date_to, location, device };
    }

    const params = { ...lastSearchParams, top_k: PAGE_SIZE, offset: searchOffset };

    if (!append) showSearchLoading();
    document.getElementById('load-more-btn')?.remove();

    fetch(`${API}/api/search`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(params),
    })
    .then(r => r.json())
    .then(data => {
        renderPhotoGrid(data, 'grid', true, append);
        renderSearchInfo(data);
        searchOffset += data.results.length;
        if (data.has_more) {
            const grid = document.getElementById('grid');
            const moreBtn = document.createElement('div');
            moreBtn.id = 'load-more-btn';
            moreBtn.className = 'load-more-btn';
            moreBtn.innerHTML = '<button onclick="doSearch(true)">加载更多 ▼</button>';
            grid.parentElement.appendChild(moreBtn);
        }
    })
    .catch(showSearchError);
}

function example(btn, opts) {
    document.getElementById('c-text').value = opts.text || '';
    document.getElementById('c-date-from').value = opts.date_from || '';
    document.getElementById('c-date-to').value = opts.date_to || '';
    document.getElementById('c-location').value = opts.location || '';
    document.getElementById('c-device').value = opts.device || '';
    doSearch(false);
}

function val(id) {
    const v = document.getElementById(id).value.trim();
    return v || null;
}

function showSearchLoading() {
    hide('empty'); hide('grid'); hide('results-info');
    show('loading');
}

function showSearchError() {
    hide('loading'); hide('grid'); hide('results-info');
    const el = document.getElementById('empty');
    el.style.display = '';
    el.innerHTML = '<p>搜索出错，请检查服务</p>';
}

function renderSearchInfo(data) {
    const info = document.getElementById('results-info');
    if (!info) return;

    const shown = searchOffset + (data.results?.length || 0);
    if (typeof data.total === 'number') {
        info.textContent = `找到 ${data.total} 张匹配照片`;
    } else {
        info.textContent = `已显示 ${shown} 张结果`;
    }
    info.style.display = '';
}

// ========== Photo Grid ==========
function renderPhotoGrid(data, gridId, showSim, append) {
    if (!append) {
        hide('loading'); hide('empty');
        const grid = document.getElementById(gridId);
        grid.innerHTML = '';
        grid.style.display = '';
    }

    const items = data.results || data.photos || [];
    const grid = document.getElementById(gridId);

    // 断崖检测：距第一张下降超过 3%
    let cliffIdx = -1;
    if (showSim && items.length > 3) {
        const firstSim = items[0].similarity || 0;
        for (let i = 1; i < items.length; i++) {
            if (firstSim - (items[i].similarity || 0) > 0.03) {
                cliffIdx = i;
                break;
            }
        }
    }

    items.forEach((p, idx) => {
        const sim = Math.round((p.similarity || 0) * 100);
        const isVideo = p.media_type === 'video';
        const dur = isVideo && p.video_duration ? formatDuration(p.video_duration) : '';

        // 断崖分隔线
        if (cliffIdx === idx) {
            const sep = document.createElement('div');
            sep.className = 'cliff-separator';
            sep.innerHTML = `<span>以下相关度较低 (${sim}%)</span>`;
            grid.appendChild(sep);
        }

        const card = document.createElement('div');
        card.className = 'card';
        card.onclick = () => openModal(p);

        // badge: 视频显示 ▶ 时长 + 相似度，照片只显示相似度
        let badge = '';
        if (isVideo) {
            badge = `<span class="badge video-badge">▶ ${dur}${showSim ? ' · ' + sim + '%' : ''}</span>`;
        } else if (showSim) {
            badge = `<span class="badge">${sim}%</span>`;
        }

        card.innerHTML = `
            <img src="${API}/api/photos/${p.id}/thumbnail" loading="lazy"
                 onerror="this.style.display='none'">
            ${badge}
            <div class="info">
                <div>${p.date_original || ''}</div>
                <div>${[p.gps_city, p.gps_country].filter(Boolean).join(', ')}</div>
                <div>${isVideo ? '🎬 视频' : [p.camera_make, p.camera_model].filter(Boolean).join(' ')}</div>
            </div>`;
        grid.appendChild(card);
    });
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}:${s.toString().padStart(2, '0')}` : `${s}s`;
}

// ========== People Page ==========
function loadPeople() {
    show('people-loading');
    hide('people-overview');
    hide('people-detail');
    hide('reindex-status');

    fetch(`${API}/api/faces/clusters`)
        .then(r => r.json())
        .then(data => {
            hide('people-loading');
            clustersCache = data.clusters || [];
            renderPeopleGrid();
            show('people-overview');
        })
        .catch(() => {
            hide('people-loading');
            show('people-overview');
            document.getElementById('people-grid').innerHTML =
                '<p style="text-align:center;color:var(--dim);padding:40px">加载失败</p>';
        });
}

function renderPeopleGrid() {
    const visible = clustersCache.filter(c => !c.hidden);
    const hidden = clustersCache.filter(c => c.hidden);

    document.getElementById('people-count').textContent = `共 ${visible.length} 人`;

    // 正常人物
    const grid = document.getElementById('people-grid');
    grid.innerHTML = '';
    visible.forEach((c, idx) => {
        grid.appendChild(createPersonCard(c, idx, false));
    });

    // 合并模式样式
    const gridEl = document.getElementById('people-grid');
    if (mergeMode) gridEl.classList.add('merge-mode');
    else gridEl.classList.remove('merge-mode');

    // 隐藏区域
    const hiddenSection = document.getElementById('hidden-section');
    if (hidden.length > 0) {
        hiddenSection.style.display = '';
        document.getElementById('hidden-count-text').textContent = `已隐藏 ${hidden.length} 人`;
        const hiddenGrid = document.getElementById('hidden-grid');
        hiddenGrid.innerHTML = '';
        hidden.forEach((c, idx) => {
            hiddenGrid.appendChild(createPersonCard(c, idx, true));
        });
    } else {
        hiddenSection.style.display = 'none';
    }
}

function createPersonCard(c, idx, isHidden) {
    const name = c.person_name || '';
    const displayName = name || `人物 ${idx + 1}`;

    const card = document.createElement('div');
    card.className = 'person-card';
    if (selectedForMerge.has(c.cluster_id)) card.classList.add('selected');
    card.dataset.clusterId = c.cluster_id;

    card.innerHTML = `
        <div class="person-check ${selectedForMerge.has(c.cluster_id) ? 'checked' : ''}"
             onclick="event.stopPropagation(); toggleMergeSelect('${c.cluster_id}')">✓</div>
        ${isHidden
            ? `<button class="person-restore-btn" onclick="event.stopPropagation(); restorePerson('${c.cluster_id}')" title="恢复显示">👁 恢复</button>`
            : `<button class="person-hide-btn" onclick="event.stopPropagation(); hidePerson('${c.cluster_id}')" title="隐藏此人">👁‍🗨</button>`
        }
        <img class="person-avatar" src="${API}/api/faces/${c.representative_id}/thumbnail"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%231c1c1c%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2255%22 text-anchor=%22middle%22 fill=%22%23777%22 font-size=%2240%22>?</text></svg>'">
        <div class="person-name ${name ? '' : 'unnamed'}">${displayName}</div>
        <div class="person-count">${c.photo_count} 张照片</div>`;

    card.onclick = (e) => {
        if (e.target.closest('.person-check') || e.target.closest('.person-hide-btn') || e.target.closest('.person-restore-btn')) return;
        if (mergeMode) {
            toggleMergeSelect(c.cluster_id);
        } else {
            showPersonDetail(c.cluster_id);
        }
    };
    return card;
}

// ========== Merge ==========
function toggleMergeMode() {
    mergeMode = !mergeMode;
    selectedForMerge.clear();

    const btn = document.getElementById('merge-mode-btn');
    const bar = document.getElementById('merge-bar');
    const grid = document.getElementById('people-grid');

    if (mergeMode) {
        btn.classList.add('active');
        bar.style.display = '';
        grid.classList.add('merge-mode');
    } else {
        btn.classList.remove('active');
        bar.style.display = 'none';
        grid.classList.remove('merge-mode');
    }
    renderPeopleGrid();
}

function toggleMergeSelect(clusterId) {
    if (selectedForMerge.has(clusterId)) {
        selectedForMerge.delete(clusterId);
    } else {
        selectedForMerge.add(clusterId);
    }
    document.getElementById('merge-selected-count').textContent = `已选 ${selectedForMerge.size} 人`;

    // 更新卡片样式
    document.querySelectorAll('.person-card').forEach(card => {
        const cid = card.dataset.clusterId;
        const check = card.querySelector('.person-check');
        if (selectedForMerge.has(cid)) {
            card.classList.add('selected');
            if (check) check.classList.add('checked');
        } else {
            card.classList.remove('selected');
            if (check) check.classList.remove('checked');
        }
    });
}

function doMerge() {
    if (selectedForMerge.size < 2) {
        alert('请至少选择 2 个人物');
        return;
    }
    const name = document.getElementById('merge-name-input').value.trim();
    const ids = Array.from(selectedForMerge);

    fetch(`${API}/api/faces/clusters/merge`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ cluster_ids: ids, name }),
    })
    .then(r => r.json())
    .then(() => {
        toggleMergeMode();
        loadPeople();
    })
    .catch(() => alert('合并失败'));
}

// ========== Hide / Restore ==========
function hidePerson(clusterId) {
    fetch(`${API}/api/faces/clusters/${clusterId}/hidden`, {
        method: 'PUT',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ hidden: true }),
    })
    .then(r => r.json())
    .then(() => {
        // 更新缓存
        const c = clustersCache.find(c => c.cluster_id === clusterId);
        if (c) c.hidden = true;
        renderPeopleGrid();
    });
}

function restorePerson(clusterId) {
    fetch(`${API}/api/faces/clusters/${clusterId}/hidden`, {
        method: 'PUT',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ hidden: false }),
    })
    .then(r => r.json())
    .then(() => {
        const c = clustersCache.find(c => c.cluster_id === clusterId);
        if (c) c.hidden = false;
        renderPeopleGrid();
    });
}

function toggleHiddenSection() {
    const grid = document.getElementById('hidden-grid');
    const icon = document.getElementById('hidden-toggle-icon');
    if (grid.style.display === 'none') {
        grid.style.display = '';
        icon.textContent = '▼';
    } else {
        grid.style.display = 'none';
        icon.textContent = '▶';
    }
}

// ========== Person Detail ==========
function showPersonDetail(clusterId) {
    currentClusterId = clusterId;
    hide('people-overview');
    show('people-detail');

    const cluster = clustersCache.find(c => c.cluster_id === clusterId);
    if (cluster) {
        document.getElementById('detail-avatar').src =
            `${API}/api/faces/${cluster.representative_id}/thumbnail`;
        document.getElementById('detail-name').textContent =
            cluster.person_name || '未命名';
        document.getElementById('detail-count').textContent =
            `${cluster.photo_count} 张照片`;
        document.getElementById('detail-name-input').value =
            cluster.person_name || '';
    }

    // 清空搜索栏筛选条件
    document.getElementById('c-date-from').value = '';
    document.getElementById('c-date-to').value = '';
    document.getElementById('c-location').value = '';
    document.getElementById('c-device').value = '';

    loadPersonPhotos(clusterId);
}

function filterPersonPhotos() {
    loadPersonPhotos(currentClusterId);
}

function loadPersonPhotos(clusterId) {
    const params = new URLSearchParams();
    const df = document.getElementById('c-date-from').value;
    const dt = document.getElementById('c-date-to').value;
    const loc = document.getElementById('c-location').value.trim();
    const dev = document.getElementById('c-device').value.trim();
    if (df) params.set('date_from', df);
    if (dt) params.set('date_to', dt);
    if (loc) params.set('location', loc);
    if (dev) params.set('device', dev);

    const qs = params.toString();
    fetch(`${API}/api/faces/clusters/${clusterId}/photos${qs ? '?' + qs : ''}`)
        .then(r => r.json())
        .then(data => {
            const count = data.photos?.length || 0;
            document.getElementById('detail-count').textContent = `${count} 张照片`;
            renderPhotoGrid(data, 'detail-grid', false, false);
        })
        .catch(() => {
            document.getElementById('detail-grid').innerHTML =
                '<p style="text-align:center;color:var(--dim);padding:40px">加载失败</p>';
        });
}

function showPeopleOverview() {
    hide('people-detail');
    show('people-overview');
}

function saveName() {
    const name = document.getElementById('detail-name-input').value.trim();
    if (!name || !currentClusterId) return;

    fetch(`${API}/api/faces/clusters/${currentClusterId}`, {
        method: 'PUT',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ name }),
    })
    .then(r => r.json())
    .then(() => {
        document.getElementById('detail-name').textContent = name;
        const c = clustersCache.find(c => c.cluster_id === currentClusterId);
        if (c) c.person_name = name;
    });
}

function reindexFaces() {
    if (!confirm('重新识别会清除所有现有标注和合并，确定？')) return;
    hide('people-overview');
    hide('people-detail');
    show('reindex-status');

    fetch(`${API}/api/faces/reindex`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            hide('reindex-status');
            alert(`识别完成：检测到 ${data.faces_detected} 张脸，分为 ${data.clusters} 个人`);
            loadPeople();
        })
        .catch(() => {
            hide('reindex-status');
            show('people-overview');
            alert('重新识别失败');
        });
}

// ========== Modal ==========
function openModal(p) {
    const modal = document.getElementById('modal');
    const img = document.getElementById('modal-img');
    const video = document.getElementById('modal-video');
    const isVideo = p.media_type === 'video';

    if (isVideo) {
        img.style.display = 'none';
        video.style.display = '';
        video.src = `${API}/api/photos/${p.id}/file`;
    } else {
        video.style.display = 'none';
        video.src = '';
        img.style.display = '';
        img.src = `${API}/api/photos/${p.id}/thumbnail`;
        const full = new Image();
        full.onload = () => { img.src = full.src; };
        full.src = `${API}/api/photos/${p.id}/file`;
    }
    document.getElementById('modal-info').textContent = [
        p.date_original,
        [p.gps_city, p.gps_country].filter(Boolean).join(', '),
        [p.camera_make, p.camera_model].filter(Boolean).join(' '),
        p.width && p.height ? `${p.width}×${p.height}` : '',
    ].filter(Boolean).join('  |  ');

    const facesEl = document.getElementById('modal-faces');
    facesEl.innerHTML = '';
    fetch(`${API}/api/faces/photos/${p.id}`)
        .then(r => r.json())
        .then(data => {
            if (data.faces && data.faces.length > 0) {
                data.faces.forEach(f => {
                    const el = document.createElement('div');
                    el.className = 'modal-face';
                    el.onclick = (e) => {
                        e.stopPropagation();
                        closeModal();
                        switchTabDirect('people');
                        setTimeout(() => showPersonDetail(f.cluster_id), 200);
                    };
                    el.innerHTML = `
                        <img src="${API}/api/faces/${f.id}/thumbnail"
                             onerror="this.style.display='none'">
                        <span class="${f.person_name ? '' : 'unnamed'}">${f.person_name || '未知'}</span>`;
                    facesEl.appendChild(el);
                });
            }
        });

    modal.classList.add('active');
}

// 直接切 tab 不依赖 event
function switchTabDirect(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.textContent.includes(tab === 'search' ? '搜索' : '人物'));
    });
    document.getElementById('content-row').style.display = tab === 'search' ? '' : 'none';
    if (tab === 'search') { show('page-search'); hide('page-people'); }
    else { hide('page-search'); show('page-people'); }
}

function closeModal(e) {
    if (e && e.target !== document.getElementById('modal') && !e.target.classList.contains('close-btn')) return;
    document.getElementById('modal').classList.remove('active');
    const video = document.getElementById('modal-video');
    video.pause();
    video.src = '';
}

// ========== Utils ==========
function show(id) { document.getElementById(id).style.display = ''; }
function hide(id) { document.getElementById(id).style.display = 'none'; }

document.addEventListener('DOMContentLoaded', () => {
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') document.getElementById('modal').classList.remove('active');
        if (e.key === 'Enter' && e.target.tagName === 'INPUT' && currentTab === 'search') doSearch(false);
    });
});
