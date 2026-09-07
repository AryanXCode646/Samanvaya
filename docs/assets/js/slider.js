/**
 * Samanvaya (समान्वय) — Interactive Before/After Split Comparison Slider
 * Precision Touch/Mouse/Keyboard Slider with Multi-Modal Image Simulation
 * ISRO SIH PS 26166
 */

(function () {
  'use strict';

  function initComparisonSliders() {
    const containers = document.querySelectorAll('.slider-container');
    if (!containers.length) return;

    containers.forEach((container) => {
      const divider = container.querySelector('.slider-divider');
      const afterLayer = container.querySelector('.slider-after');
      const beforeBadge = container.querySelector('.slider-badge-before');
      const afterBadge = container.querySelector('.slider-badge-after');

      if (!divider || !afterLayer) return;

      let isDragging = false;
      let currentPercent = 50;

      function setPosition(percent) {
        // Clamp between 2% and 98%
        const clamped = Math.max(2, Math.min(98, percent));
        currentPercent = clamped;

        divider.style.left = `${clamped}%`;
        afterLayer.style.clipPath = `inset(0 0 0 ${clamped}%)`;
      }

      function onMove(clientX) {
        const rect = container.getBoundingClientRect();
        const offsetX = clientX - rect.left;
        const percent = (offsetX / rect.width) * 100;
        setPosition(percent);
      }

      // Mouse events
      container.addEventListener('mousedown', (e) => {
        isDragging = true;
        onMove(e.clientX);
      });

      window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        onMove(e.clientX);
      });

      window.addEventListener('mouseup', () => {
        isDragging = false;
      });

      // Touch events for mobile/tablet
      container.addEventListener(
        'touchstart',
        (e) => {
          isDragging = true;
          if (e.touches && e.touches[0]) {
            onMove(e.touches[0].clientX);
          }
        },
        { passive: true }
      );

      window.addEventListener(
        'touchmove',
        (e) => {
          if (!isDragging) return;
          if (e.touches && e.touches[0]) {
            onMove(e.touches[0].clientX);
          }
        },
        { passive: true }
      );

      window.addEventListener('touchend', () => {
        isDragging = false;
      });

      // Keyboard accessibility (Left / Right arrow keys)
      container.setAttribute('tabindex', '0');
      container.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          setPosition(currentPercent - 5);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          setPosition(currentPercent + 5);
        }
      });

      // Mode switching support
      const modeButtons = document.querySelectorAll('[data-slider-mode]');
      modeButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
          const mode = btn.getAttribute('data-slider-mode');
          modeButtons.forEach((b) => b.classList.remove('active', 'border-cyan-400', 'text-cyan-300', 'bg-cyan-950/60'));
          btn.classList.add('active', 'border-cyan-400', 'text-cyan-300', 'bg-cyan-950/60');

          const beforeImg = container.querySelector('.slider-before img');
          const afterImg = container.querySelector('.slider-after img');

          if (!beforeImg || !afterImg) return;

          if (mode === 'shadow') {
            beforeImg.src = 'assets/hero_banner.png';
            afterImg.src = 'assets/proof_in_3_seconds.png';
            if (beforeBadge) beforeBadge.textContent = 'OHRC Source (Azimuth: 60° Morning)';
            if (afterBadge) afterBadge.textContent = 'LRO NAC Reference (Azimuth: 240° Afternoon)';
          } else if (mode === 'pc') {
            beforeImg.src = 'assets/visual_phase_congruency.png';
            afterImg.src = 'assets/proof_in_3_seconds.png';
            if (beforeBadge) beforeBadge.textContent = 'Vectorized Log-Gabor Phase Congruency (M_max)';
            if (afterBadge) afterBadge.textContent = 'Sub-Pixel Aligned Physical Crests';
          } else if (mode === 'checkerboard') {
            beforeImg.src = 'assets/proof_in_3_seconds.png';
            afterImg.src = 'assets/sample_residual_scatter.png';
            if (beforeBadge) beforeBadge.textContent = '50/50 Mosaic Checkerboard Fusion';
            if (afterBadge) afterBadge.textContent = 'Sub-Pixel Residuals (RMSE: 0.338 px)';
          }
        });
      });

      // Subtle initial glide animation
      setTimeout(() => {
        let step = 0;
        const glideInterval = setInterval(() => {
          step++;
          const t = step / 30;
          // Smooth sine oscillation between 45% and 55%
          const p = 50 + Math.sin(t * Math.PI) * 8;
          setPosition(p);
          if (step >= 30) {
            clearInterval(glideInterval);
            setPosition(50);
          }
        }, 20);
      }, 500);
    });
  }

  // Auto-initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initComparisonSliders);
  } else {
    initComparisonSliders();
  }

  // Expose globally
  window.initComparisonSliders = initComparisonSliders;
})();
