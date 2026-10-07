/**
 * Translations replaced in browser tests: every key renders as itself, followed by its interpolation values, so a
 * test reads which message was shown and with what. Mock with `vi.mock('react-i18next', () =>
 * import('./clipTagsI18n.testing'))`.
 */
export const useTranslation = () => ({
  i18n: { resolvedLanguage: 'en' },
  t: (key: string, options?: Record<string, unknown>) =>
    options && Object.keys(options).length > 0
      ? `${key}(${Object.entries(options)
          .map(([name, value]) => `${name}=${String(value)}`)
          .join(',')})`
      : key,
});
