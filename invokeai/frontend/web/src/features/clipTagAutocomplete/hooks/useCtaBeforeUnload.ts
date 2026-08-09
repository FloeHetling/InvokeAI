import { useEffect } from 'react';

/**
 * Best-effort guard against accidental page close during CTA write operations.
 *
 * This is a browser-level warning only — it does not guarantee the operation
 * survives browser termination.
 */
export function useCtaBeforeUnload(isActive: boolean): void {
  useEffect(() => {
    if (!isActive) {
      return;
    }

    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = '';
    };

    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [isActive]);
}
