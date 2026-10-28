/**
 * FR2000 Web Dashboard - Lógica de Control y Telemetría en Tiempo Real
 * Pipeline: YuNet Face Detection -> FaceTracker -> AttentionEngine -> SFace Recognition -> Autonomous Greeting
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Elementos del DOM
  const streamImg = document.getElementById("streamImg");
  const stopBtn = document.getElementById("stopBtn");
  const pauseOverlay = document.getElementById("pauseOverlay");
  const modeButtons = document.querySelectorAll(".mode-btn");

  // Sujeto Principal
  const subjectMetaLabel = document.getElementById("subjectMetaLabel");
  const subjectStatusBadge = document.getElementById("subjectStatusBadge");
  const subjectName = document.getElementById("subjectName");
  const subjectSimilarity = document.getElementById("subjectSimilarity");
  const subjectPersonsList = document.getElementById("subjectPersonsList");
  const subjectAttentionScore = document.getElementById("subjectAttentionScore");
  const subjectCooldown = document.getElementById("subjectCooldown");
  const progressFill = document.getElementById("progressFill");

  // Métricas Grid
  const fpsVisualEl = document.getElementById("valFpsVisual");
  const fpsInfEl = document.getElementById("valFpsInf");
  const latencyEl = document.getElementById("valLatency");
  const rostrosEl = document.getElementById("valRostros");
  const identidadesEl = document.getElementById("valIdentidades");
  const audioValEl = document.getElementById("valAudio");

  // Controles
  const scoreSlider = document.getElementById("scoreSlider");
  const scoreSliderVal = document.getElementById("scoreSliderVal");
  const simSlider = document.getElementById("simSlider");
  const simSliderVal = document.getElementById("simSliderVal");
  const audioToggle = document.getElementById("audioToggle");
  const btnReload = document.getElementById("btnReload");
  const btnSnapshot = document.getElementById("btnSnapshot");

  // Catálogo de Identidades y Eventos
  const identitiesChips = document.getElementById("identitiesChips");
  const eventsList = document.getElementById("eventsList");

  // Canvas Gráfico
  const chartCanvas = document.getElementById("attentionChart");
  const ctx = chartCanvas ? chartCanvas.getContext("2d") : null;

  let isPaused = false;
  let historyData = [];
  let slidersInitialized = false;

  // =========================================================================
  // 2. CANVAS: RENDERIZADOR DEL HISTORIAL DE ATENCIÓN FACIAL
  // =========================================================================
  function resizeCanvas() {
    if (!chartCanvas) return;
    const rect = chartCanvas.parentElement.getBoundingClientRect();
    chartCanvas.width = rect.width * window.devicePixelRatio;
    chartCanvas.height = rect.height * window.devicePixelRatio;
    if (ctx) ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function renderChart() {
    if (!ctx || !chartCanvas) return;
    const rect = chartCanvas.parentElement.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    // Fondo sutil de cuadrícula horizontal
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let y = 0.25; y <= 0.75; y += 0.25) {
      ctx.beginPath();
      ctx.moveTo(0, h * y);
      ctx.lineTo(w, h * y);
      ctx.stroke();
    }

    // Línea de umbral dorado a 60% (Umbral ENGAGING de FR2000)
    const thresholdY = h * (1.0 - 0.60);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(0, thresholdY);
    ctx.lineTo(w, thresholdY);
    ctx.stroke();

    // Etiqueta del umbral
    ctx.fillStyle = "rgba(245, 158, 11, 0.75)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText("60% UMBRAL ENGAGING", w - 8, thresholdY - 4);
    ctx.restore();

    if (!historyData || historyData.length < 2) return;

    // Puntos continuos
    const stepX = w / (historyData.length - 1);
    const points = historyData.map((d, i) => {
      const val = Math.max(0, Math.min(100, d.attention || 0));
      return {
        x: i * stepX,
        y: h - (val / 100) * (h * 0.85) - 4
      };
    });

    // 1. Área con gradiente debajo de la curva
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(0, 240, 255, 0.28)");
    grad.addColorStop(0.7, "rgba(0, 240, 255, 0.05)");
    grad.addColorStop(1, "rgba(0, 240, 255, 0.00)");

    ctx.beginPath();
    ctx.moveTo(points[0].x, h);
    ctx.lineTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const xc = (points[i].x + points[i - 1].x) / 2;
      const yc = (points[i].y + points[i - 1].y) / 2;
      ctx.quadraticCurveTo(points[i - 1].x, points[i - 1].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // 2. Trazo superior luminoso de la curva
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const xc = (points[i].x + points[i - 1].x) / 2;
      const yc = (points[i].y + points[i - 1].y) / 2;
      ctx.quadraticCurveTo(points[i - 1].x, points[i - 1].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);

    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2.2;
    ctx.shadowColor = "rgba(0, 240, 255, 0.6)";
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // =========================================================================
  // 3. SINCRONIZACIÓN DE TELEMETRÍA (POLLING 250ms)
  // =========================================================================
  async function fetchTelemetry() {
    try {
      const res = await fetch("/api/telemetry");
      if (!res.ok) return;
      const data = await res.json();
      updateDashboardUI(data);
    } catch (e) {
      console.warn("Falla temporal de telemetría:", e);
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/[&<>"']/g, s => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[s]));
  }

  function updateDashboardUI(payload) {
    const t = payload.telemetry || {};
    historyData = payload.history || [];

    // A. Actualizar Sujeto Principal y Personas Reconocidas
    const primary = t.primary_subject || {};
    const attScore = t.attention_score || 0.0;
    const recognizedPersons = primary.recognized_persons || [];

    if (recognizedPersons.length > 1) {
      if (subjectMetaLabel) {
        subjectMetaLabel.textContent = `PERSONAS RECONOCIDAS (${recognizedPersons.length})`;
      }
      if (subjectName) {
        subjectName.textContent = primary.name || "Grupo Reconocido";
        subjectName.classList.add("multi-person");
      }
      if (subjectSimilarity) {
        subjectSimilarity.textContent = `${primary.similarity}% Promedio`;
        subjectSimilarity.style.color = "var(--accent-green)";
        subjectSimilarity.style.borderColor = "rgba(16, 185, 129, 0.4)";
        subjectSimilarity.style.background = "rgba(16, 185, 129, 0.12)";
      }

      // Renderizar chips para cada persona reconocida
      if (subjectPersonsList) {
        subjectPersonsList.style.display = "flex";
        subjectPersonsList.innerHTML = recognizedPersons.map(p => {
          const cdClass = p.in_cooldown ? "cooldown" : "available";
          const cdLabel = p.cooldown || "Disponible";
          return `
            <div class="subject-person-chip">
              <div class="chip-name">
                <span class="chip-dot"></span>
                <span>${escapeHtml(p.name)}</span>
              </div>
              <div class="chip-meta">
                <span class="chip-sim">${p.similarity}%</span>
                <span class="chip-cd ${cdClass}">${escapeHtml(cdLabel)}</span>
              </div>
            </div>
          `;
        }).join("");
      }
    } else {
      if (subjectMetaLabel) {
        subjectMetaLabel.textContent = "SUJETO DETECTADO";
      }
      if (subjectName) {
        subjectName.textContent = primary.name || "En Espera";
        subjectName.classList.remove("multi-person");
      }
      if (subjectPersonsList) {
        subjectPersonsList.style.display = "none";
        subjectPersonsList.innerHTML = "";
      }

      if (subjectSimilarity) {
        if (primary.decision === "KNOWN") {
          subjectSimilarity.textContent = `${primary.similarity}% Similitud`;
          subjectSimilarity.style.color = "var(--accent-green)";
          subjectSimilarity.style.borderColor = "rgba(16, 185, 129, 0.4)";
          subjectSimilarity.style.background = "rgba(16, 185, 129, 0.12)";
        } else if (primary.decision === "UNKNOWN") {
          subjectSimilarity.textContent = `${primary.similarity}% Visitante`;
          subjectSimilarity.style.color = "var(--accent-amber)";
          subjectSimilarity.style.borderColor = "rgba(245, 158, 11, 0.4)";
          subjectSimilarity.style.background = "rgba(245, 158, 11, 0.12)";
        } else if (primary.decision === "IDENTIFYING" || primary.name === "Identificando...") {
          subjectSimilarity.textContent = `Identificando...`;
          subjectSimilarity.style.color = "var(--accent-cyan)";
          subjectSimilarity.style.borderColor = "rgba(0, 229, 255, 0.4)";
          subjectSimilarity.style.background = "rgba(0, 229, 255, 0.10)";
        } else {
          subjectSimilarity.textContent = `--% Similitud`;
          subjectSimilarity.style.color = "var(--text-muted)";
          subjectSimilarity.style.borderColor = "rgba(255, 255, 255, 0.1)";
          subjectSimilarity.style.background = "rgba(255, 255, 255, 0.04)";
        }
      }
    }

    if (subjectAttentionScore) {
      subjectAttentionScore.textContent = `${attScore.toFixed(1)}%`;
    }

    if (subjectCooldown) {
      const cd = primary.cooldown || "Disponible";
      subjectCooldown.textContent = cd;
      if (cd.includes("Cooldown") || cd.includes("Parcial")) {
        subjectCooldown.style.color = "var(--accent-amber)";
      } else {
        subjectCooldown.style.color = "var(--accent-green)";
      }
    }

    if (progressFill) {
      progressFill.style.width = Math.min(100, Math.max(0, attScore)) + "%";
    }

    if (subjectStatusBadge) {
      const state = primary.state || "ESPERA";
      subjectStatusBadge.textContent = state;
      subjectStatusBadge.className = "subject-status-pill";
      if (state === "ENGAGED" || state === "ENGAGING") {
        subjectStatusBadge.classList.add("engaging");
      } else if (state === "CANDIDATE") {
        subjectStatusBadge.classList.add("candidate");
      } else {
        subjectStatusBadge.classList.add("waiting");
      }
    }

    // B. Cuadrícula de métricas secundarias
    if (fpsVisualEl) fpsVisualEl.textContent = t.fps_visual || "30.0";
    if (fpsInfEl) fpsInfEl.textContent = t.fps_detection || "28.5";
    if (latencyEl) latencyEl.innerHTML = `${t.latency_ms || 42.0} <span class="tile-unit">ms</span>`;
    if (rostrosEl) {
      const active = t.active_tracks_count || 0;
      rostrosEl.innerHTML = `${active} <span class="tile-unit">en escena</span>`;
    }
    if (identidadesEl && t.registered_identities) {
      identidadesEl.innerHTML = `${t.registered_identities.length} <span class="tile-unit">registradas</span>`;
    }
    if (audioValEl) {
      audioValEl.innerHTML = `${t.last_greeting_latency || "0.2 ms"}`;
    }

    // C. Renderizar chips de identidades registradas
    if (identitiesChips && t.registered_identities) {
      renderIdentitiesChips(t.registered_identities);
    }

    // D. Sincronizar sliders en primera carga
    if (!slidersInitialized && t.score_threshold) {
      const scPct = Math.round(t.score_threshold * 100);
      if (scoreSlider) scoreSlider.value = scPct;
      if (scoreSliderVal) scoreSliderVal.textContent = scPct + "%";
      if (t.similarity_threshold) {
        const simPct = Math.round(t.similarity_threshold * 100);
        if (simSlider) simSlider.value = simPct;
        if (simSliderVal) simSliderVal.textContent = simPct + "%";
      }
      slidersInitialized = true;
    }

    // E. Registro de eventos recientes
    if (eventsList && payload.events) {
      renderEvents(payload.events);
    }

    // F. Redibujar el gráfico
    renderChart();
  }

  function renderIdentitiesChips(identities) {
    if (!identities || identities.length === 0) {
      identitiesChips.innerHTML = `<span class="identity-chip">Sin identidades</span>`;
      return;
    }
    identitiesChips.innerHTML = identities.map(id => `
      <span class="identity-chip" title="ID: ${id.id} - ${id.samples} muestras con normalización fotométrica">
        <span class="chip-dot"></span>
        <span>${id.name}</span>
        <span class="chip-samples">(${id.samples})</span>
        <span class="chip-delete" data-id="${id.id}" data-name="${id.name}" title="Eliminar de la base de datos">×</span>
      </span>
    `).join("");
  }

  function renderEvents(events) {
    eventsList.innerHTML = "";
    events.slice(0, 10).forEach(ev => {
      const li = document.createElement("li");
      li.className = "event-item";

      const catLower = (ev.category || "info").toLowerCase();
      li.innerHTML = `
        <span class="event-time">${ev.timestamp}</span>
        <span class="event-badge ${catLower}">${ev.category}</span>
        <span class="event-text" title="${ev.text}">${ev.text}</span>
      `;
      eventsList.appendChild(li);
    });
  }

  // =========================================================================
  // 4. INTERACTIVIDAD DE CONTROLES
  // =========================================================================

  // Botón STOP / START
  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/controls/toggle_pause", { method: "POST" });
        const data = await res.json();
        isPaused = data.is_paused;
        if (isPaused) {
          stopBtn.textContent = "▶ REANUDAR";
          stopBtn.classList.add("paused");
          pauseOverlay.classList.add("visible");
        } else {
          stopBtn.textContent = "🛑 PAUSAR";
          stopBtn.classList.remove("paused");
          pauseOverlay.classList.remove("visible");
        }
      } catch (e) {
        console.error("Fallo al alternar pausa:", e);
      }
    });
  }

  // Selector de Modo de Visualización (HUD Biométrico, Enfoque Facial, Cámara Limpia)
  modeButtons.forEach(btn => {
    btn.addEventListener("click", async () => {
      modeButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.dataset.mode;
      try {
        await fetch("/api/controls/view_mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode })
        });
      } catch (e) {
        console.error("Fallo al cambiar modo:", e);
      }
    });
  });

  // Slider de Sensibilidad YuNet
  if (scoreSlider) {
    scoreSlider.addEventListener("input", (e) => {
      scoreSliderVal.textContent = e.target.value + "%";
    });

    scoreSlider.addEventListener("change", async (e) => {
      const th = parseFloat(e.target.value) / 100.0;
      try {
        await fetch("/api/controls/threshold", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold: th })
        });
      } catch (e) {
        console.error("Fallo al calibrar sensibilidad YuNet:", e);
      }
    });
  }

  // Slider de Similitud Facial SFace
  if (simSlider) {
    simSlider.addEventListener("input", (e) => {
      simSliderVal.textContent = e.target.value + "%";
    });

    simSlider.addEventListener("change", async (e) => {
      const th = parseFloat(e.target.value) / 100.0;
      try {
        await fetch("/api/controls/similarity_threshold", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold: th })
        });
      } catch (e) {
        console.error("Fallo al calibrar similitud SFace:", e);
      }
    });
  }

  // Toggle Saludos de Voz
  if (audioToggle) {
    audioToggle.addEventListener("change", async (e) => {
      const enabled = e.target.checked;
      try {
        await fetch("/api/controls/toggle_audio", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled })
        });
      } catch (e) {
        console.error("Fallo al alternar audio:", e);
      }
    });
  }

  // Botón Recargar Identidades
  if (btnReload) {
    btnReload.addEventListener("click", async () => {
      btnReload.disabled = true;
      btnReload.textContent = "Recargando...";
      try {
        const res = await fetch("/api/actions/reload_identities", { method: "POST" });
        const data = await res.json();
        alert(`✓ Catálogo biométrico recargado: ${data.embeddings_count} firmas en memoria.`);
      } catch (e) {
        alert("✗ Error recargando identidades.");
      } finally {
        btnReload.disabled = false;
        btnReload.innerHTML = `<span>⟳</span> Recargar DB`;
      }
    });
  }

  // Botón Capturar Snapshot
  if (btnSnapshot) {
    btnSnapshot.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/actions/snapshot", { method: "POST" });
        const data = await res.json();
        if (data.filepath) {
          alert(`✓ Captura de diagnóstico guardada en: ${data.filepath}`);
        }
      } catch (e) {
        alert("✗ Error guardando snapshot.");
      }
    });
  }

  // =========================================================================
  // 5. REGISTRO DE IDENTIDADES (CÁMARA / FOTO CON NORMALIZACIÓN)
  // =========================================================================
  const tabBtnCamera = document.getElementById("tabBtnCamera");
  const tabBtnUpload = document.getElementById("tabBtnUpload");
  const panelCamera = document.getElementById("panelCamera");
  const panelUpload = document.getElementById("panelUpload");

  const regCameraName = document.getElementById("regCameraName");
  const btnRegisterCamera = document.getElementById("btnRegisterCamera");

  const regUploadName = document.getElementById("regUploadName");
  const regUploadFile = document.getElementById("regUploadFile");
  const fileDropText = document.getElementById("fileDropText");
  const btnRegisterUpload = document.getElementById("btnRegisterUpload");

  const regStatusMsg = document.getElementById("regStatusMsg");
  
  // Elementos del Modal
  const btnOpenCaptureModal = document.getElementById("btnOpenCaptureModal");
  const captureModal = document.getElementById("captureModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const modalVideoFeed = document.getElementById("modalVideoFeed");
  const modalTargetBox = document.getElementById("modalTargetBox");
  const modalRegStatus = document.getElementById("modalRegStatus");
  const step1Capture = document.getElementById("step1Capture");
  const step2Save = document.getElementById("step2Save");
  const btnCapturePhoto = document.getElementById("btnCapturePhoto");
  const btnRetakePhoto = document.getElementById("btnRetakePhoto");
  const btnSavePhoto = document.getElementById("btnSavePhoto");
  
  let capturedBlob = null;

  function showRegStatus(type, message) {
    if (!regStatusMsg) return;
    regStatusMsg.style.display = "block";
    regStatusMsg.className = `reg-status-msg ${type}`;
    regStatusMsg.textContent = message;
  }

  function hideRegStatus() {
    if (!regStatusMsg) return;
    regStatusMsg.style.display = "none";
  }

  // Alternar pestañas
  if (tabBtnCamera && tabBtnUpload) {
    tabBtnCamera.addEventListener("click", () => {
      tabBtnCamera.classList.add("active");
      tabBtnUpload.classList.remove("active");
      panelCamera.style.display = "flex";
      panelUpload.style.display = "none";
      hideRegStatus();
    });

    tabBtnUpload.addEventListener("click", () => {
      tabBtnUpload.classList.add("active");
      tabBtnCamera.classList.remove("active");
      panelUpload.style.display = "flex";
      panelCamera.style.display = "none";
      hideRegStatus();
    });
  }

  // Mostrar nombre de archivo seleccionado
  if (regUploadFile && fileDropText) {
    regUploadFile.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        fileDropText.textContent = `Archivo: ${file.name} (${Math.round(file.size / 1024)} KB)`;
      } else {
        fileDropText.textContent = "Seleccionar foto (JPG / PNG)";
      }
    });
  }

  // Lógica del Modal de Captura
  function showModalRegStatus(type, message) {
    if (!modalRegStatus) return;
    modalRegStatus.style.display = "block";
    modalRegStatus.className = `reg-status-msg ${type}`;
    modalRegStatus.textContent = message;
  }

  function hideModalRegStatus() {
    if (!modalRegStatus) return;
    modalRegStatus.style.display = "none";
  }

  function resetModalState() {
    if (step1Capture) step1Capture.style.display = "flex";
    if (step2Save) step2Save.style.display = "none";
    if (modalTargetBox) modalTargetBox.style.display = "block";
    if (modalVideoFeed) modalVideoFeed.src = "/video_feed";
    if (regCameraName) regCameraName.value = "";
    capturedBlob = null;
    hideModalRegStatus();
  }

  if (btnOpenCaptureModal) {
    btnOpenCaptureModal.addEventListener("click", async () => {
      captureModal.style.display = "flex";
      resetModalState();
      if (streamImg) streamImg.style.opacity = "0.15";
      try {
        await fetch("/api/controls/view_mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: "clean" })
        });
      } catch(e){}
    });
  }

  if (closeModalBtn) {
    closeModalBtn.addEventListener("click", async () => {
      captureModal.style.display = "none";
      if (modalVideoFeed) modalVideoFeed.src = "";
      if (streamImg) streamImg.style.opacity = "1";
      try {
        const activeMode = document.querySelector(".mode-btn.active")?.dataset.mode || "biometric";
        await fetch("/api/controls/view_mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: activeMode })
        });
      } catch(e){}
    });
  }

  if (btnCapturePhoto) {
    btnCapturePhoto.addEventListener("click", () => {
      if (!modalVideoFeed) return;
      
      const canvas = document.createElement("canvas");
      canvas.width = modalVideoFeed.naturalWidth || 1920;
      canvas.height = modalVideoFeed.naturalHeight || 1080;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(modalVideoFeed, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob((blob) => {
        capturedBlob = blob;
        modalVideoFeed.src = canvas.toDataURL("image/jpeg");
        if (modalTargetBox) modalTargetBox.style.display = "none";
        
        step1Capture.style.display = "none";
        step2Save.style.display = "flex";
        if (regCameraName) regCameraName.focus();
      }, "image/jpeg", 0.95);
    });
  }

  if (btnRetakePhoto) {
    btnRetakePhoto.addEventListener("click", () => {
      resetModalState();
    });
  }

  if (btnSavePhoto) {
    btnSavePhoto.addEventListener("click", async () => {
      const name = regCameraName ? regCameraName.value.trim() : "";
      if (!name) {
        showModalRegStatus("error", "⚠️ Por favor, ingresa el nombre de la persona.");
        if (regCameraName) regCameraName.focus();
        return;
      }
      if (!capturedBlob) {
        showModalRegStatus("error", "⚠️ No hay foto capturada.");
        return;
      }

      btnSavePhoto.disabled = true;
      showModalRegStatus("loading", `Analizando calidad de la foto (nitidez, rostro) y registrando a '${name}'...`);

      const formData = new FormData();
      formData.append("name", name);
      formData.append("file", capturedBlob, "capture.jpg");

      try {
        const res = await fetch("/api/identities/register_upload", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (data.success) {
          showModalRegStatus("success", `✓ ${data.message}`);
          await fetchTelemetry();
          setTimeout(() => {
            if (closeModalBtn) closeModalBtn.click();
          }, 2000);
        } else {
          showModalRegStatus("error", `✗ ${data.message || "Error al procesar la foto."}`);
        }
      } catch (err) {
        showModalRegStatus("error", `✗ Error de conexión con el servidor: ${err.message}`);
      } finally {
        btnSavePhoto.disabled = false;
      }
    });
  }

  // Registro desde Foto Subida
  if (btnRegisterUpload) {
    btnRegisterUpload.addEventListener("click", async () => {
      const name = regUploadName ? regUploadName.value.trim() : "";
      const file = regUploadFile && regUploadFile.files ? regUploadFile.files[0] : null;

      if (!name) {
        showRegStatus("error", "⚠️ Por favor, ingresa el nombre de la persona.");
        if (regUploadName) regUploadName.focus();
        return;
      }
      if (!file) {
        showRegStatus("error", "⚠️ Por favor, selecciona un archivo de imagen (JPG/PNG).");
        return;
      }

      btnRegisterUpload.disabled = true;
      showRegStatus("loading", `Analizando imagen, validando rostro y normalizando iluminación para '${name}'...`);

      const formData = new FormData();
      formData.append("name", name);
      formData.append("file", file);

      try {
        const res = await fetch("/api/identities/register_upload", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (data.success) {
          showRegStatus("success", `✓ ${data.message}`);
          if (regUploadName) regUploadName.value = "";
          if (regUploadFile) regUploadFile.value = "";
          if (fileDropText) fileDropText.textContent = "Seleccionar foto (JPG / PNG)";
          await fetchTelemetry();
        } else {
          showRegStatus("error", `✗ ${data.message || "Error al procesar la foto."}`);
        }
      } catch (err) {
        showRegStatus("error", `✗ Error de conexión con el servidor: ${err.message}`);
      } finally {
        btnRegisterUpload.disabled = false;
      }
    });
  }

  // Eliminación de identidad desde chip
  if (identitiesChips) {
    identitiesChips.addEventListener("click", async (e) => {
      const delBtn = e.target.closest(".chip-delete");
      if (!delBtn) return;
      const personId = delBtn.dataset.id;
      const personName = delBtn.dataset.name;
      if (!personId) return;

      const confirmDel = confirm(`¿Estás seguro de que deseas eliminar a '${personName}' de la base biométrica?`);
      if (!confirmDel) return;

      try {
        const res = await fetch(`/api/identities/${personId}`, { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
          await fetchTelemetry();
        } else {
          alert(`Error: ${data.message}`);
        }
      } catch (err) {
        alert(`Error eliminando identidad: ${err.message}`);
      }
    });
  }

  // Iniciar sondeo continuo
  setInterval(fetchTelemetry, 250);
  fetchTelemetry();
});
