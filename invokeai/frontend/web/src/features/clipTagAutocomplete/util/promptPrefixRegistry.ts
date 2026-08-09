/**
 * Prompt prefix ownership registry.
 *
 * Tracks which prompt feature owns each prefix character.
 * Used to detect hot-prefix conflicts when enabling CTA.
 */

type PromptPrefixOwner = 'promptTrigger' | 'clipTagAutocomplete';

const STATIC_PROMPT_PREFIXES = new Map<string, PromptPrefixOwner>([['<', 'promptTrigger']]);

/**
 * Returns the owner of a given prefix character, or null if unclaimed.
 */
export const getPromptPrefixOwner = (prefix: string): PromptPrefixOwner | null => {
  return STATIC_PROMPT_PREFIXES.get(prefix) ?? null;
};
