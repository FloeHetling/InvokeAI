import { useMountEffect } from '@platform/react/useMountEffect';

/**
 * Asks the browser to confirm before the tab closes or reloads. Mount it only while a write is in flight: it is
 * a best-effort warning, and the write may still be lost if the browser is closed anyway.
 */
export const BeforeUnloadGuard = () => {
  useMountEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  });

  return null;
};
