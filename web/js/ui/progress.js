// A thin progress bar along the top of the image area, shared by every long-running task.
//
// Tasks register themselves and report a fraction (0…1) or nothing (unknown duration). The bar
// shows the average of the tasks that report a fraction, or an indeterminate sweep if none do.
// It only appears once something has taken longer than SHOW_AFTER ms, so quick operations
// (most slider moves) never make it flicker.

const SHOW_AFTER = 200;

export function createProgressBar(el) {
  const fill = el.firstElementChild;
  const tasks = new Set();
  let showTimer = null;

  function render() {
    if (!tasks.size) {
      clearTimeout(showTimer);
      showTimer = null;
      el.classList.remove('active');
      el.removeAttribute('aria-valuenow');
      return;
    }
    if (!el.classList.contains('active') && !showTimer) {
      showTimer = setTimeout(() => {
        showTimer = null;
        if (tasks.size) el.classList.add('active');
      }, SHOW_AFTER);
    }
    const known = [...tasks].filter((t) => t.value != null);
    const labels = [...tasks].map((t) => t.label).join(' · ');
    el.title = labels;
    el.setAttribute('aria-valuetext', labels);
    if (known.length) {
      const value = known.reduce((sum, t) => sum + t.value, 0) / known.length;
      el.classList.remove('indeterminate');
      fill.style.width = `${Math.round(value * 100)}%`;
      el.setAttribute('aria-valuenow', String(Math.round(value * 100)));
    } else {
      el.classList.add('indeterminate');
      fill.style.width = '';
      el.removeAttribute('aria-valuenow');
    }
  }

  return {
    /** Start a task. Call `.set(fraction)` as it advances and `.done()` when finished. */
    start(label) {
      const task = { label, value: null };
      tasks.add(task);
      render();
      return {
        set(value) {
          task.value = Math.min(1, Math.max(0, value));
          render();
        },
        done() {
          tasks.delete(task);
          render();
        },
      };
    },

    /** Run `fn(onProgress)` as a task. */
    async track(label, fn) {
      const task = this.start(label);
      try {
        return await fn((p) => task.set(p));
      } finally {
        task.done();
      }
    },
  };
}
