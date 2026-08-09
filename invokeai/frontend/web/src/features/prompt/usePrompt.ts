import { useAppSelector } from 'app/store/storeHooks';
import { isNil } from 'es-toolkit/compat';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import {
  selectClipTagAutocompleteEnabled,
  selectClipTagAutocompleteHotPrefix,
} from 'features/system/store/systemSlice';
import type { ChangeEventHandler, KeyboardEventHandler, RefObject } from 'react';
import { useCallback, useRef, useState } from 'react';
import { flushSync } from 'react-dom';

type UseInsertTriggerArg = {
  prompt: string;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  onChange: (value: string) => void;
  isDisabled?: boolean;
};

type PromptAssistMode = 'promptTrigger' | 'cta' | null;

const isCtaLauncherBoundary = (prompt: string, caret: number): boolean => {
  if (caret === 0) {
    return true;
  }
  const previousCharacter = prompt[caret - 1];
  return !/[\p{L}\p{N}\p{M}_]/u.test(previousCharacter ?? '');
};

export const computeCtaInsertion = (prompt: string, caret: number, renderedContent: string) => {
  const before = prompt.slice(0, caret);
  const right = prompt.slice(caret);
  const separatorMatch = right.match(/^[ \t]*,[ \t]*/);
  const cleanedRight = separatorMatch ? right.slice(separatorMatch[0].length) : right;
  const inserted = `${renderedContent}, `;

  return {
    value: `${before}${inserted}${cleanedRight}`,
    caret: before.length + inserted.length,
  };
};

export const usePrompt = ({ prompt, textareaRef, onChange: updatePrompt, isDisabled = false }: UseInsertTriggerArg) => {
  const ctaEnabled = useAppSelector(selectClipTagAutocompleteEnabled);
  const ctaHotPrefix = useAppSelector(selectClipTagAutocompleteHotPrefix);
  const modelId = useAppSelector(selectModel)?.key ?? null;
  const [assistMode, setAssistMode] = useState<PromptAssistMode>(null);
  const ctaCaretRef = useRef<number | null>(null);

  const onChange: ChangeEventHandler<HTMLTextAreaElement | null> = useCallback(
    (event) => updatePrompt(event.target.value),
    [updatePrompt]
  );

  const insertTrigger = useCallback(
    (value: string) => {
      const textarea = textareaRef.current;
      if (!textarea) {
        return;
      }
      const caret = textarea.selectionStart;
      if (isNil(caret)) {
        return;
      }

      const newPrompt = `${prompt.slice(0, caret)}${value}${prompt.slice(caret)}`;
      const finalCaret = caret + value.length;

      flushSync(() => updatePrompt(newPrompt));
      textarea.selectionStart = finalCaret;
      textarea.selectionEnd = finalCaret;
    },
    [prompt, textareaRef, updatePrompt]
  );

  const onFocus = useCallback(() => {
    textareaRef.current?.focus();
  }, [textareaRef]);

  const onClose = useCallback(() => {
    ctaCaretRef.current = null;
    setAssistMode(null);
  }, []);

  const onOpen = useCallback(() => {
    ctaCaretRef.current = null;
    setAssistMode('promptTrigger');
  }, []);

  const onSelect = useCallback(
    (value: string) => {
      insertTrigger(value);
      setAssistMode(null);
      onFocus();
    },
    [insertTrigger, onFocus]
  );

  const onCtaSelect = useCallback(
    (renderedContent: string, keepOpen = false) => {
      const textarea = textareaRef.current;
      const caret = ctaCaretRef.current;
      if (!textarea || caret === null) {
        return;
      }

      const result = computeCtaInsertion(prompt, caret, renderedContent);
      ctaCaretRef.current = keepOpen ? result.caret : null;

      flushSync(() => {
        updatePrompt(result.value);
        if (!keepOpen) {
          setAssistMode(null);
        }
      });

      if (keepOpen) {
        return;
      }

      textarea.focus();
      textarea.selectionStart = result.caret;
      textarea.selectionEnd = result.caret;
    },
    [prompt, textareaRef, updatePrompt]
  );

  const onKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = useCallback(
    (event) => {
      if (event.key === '<' && !isDisabled) {
        ctaCaretRef.current = null;
        setAssistMode('promptTrigger');
        event.preventDefault();
        return;
      }

      if (!ctaEnabled || ctaHotPrefix === null || event.key !== ctaHotPrefix || isDisabled) {
        return;
      }

      const textarea = textareaRef.current;
      if (!textarea || textarea.selectionStart !== textarea.selectionEnd) {
        return;
      }

      const caret = textarea.selectionStart;
      if (isNil(caret) || !isCtaLauncherBoundary(prompt, caret)) {
        return;
      }

      ctaCaretRef.current = caret;
      setAssistMode('cta');
      event.preventDefault();
    },
    [ctaEnabled, ctaHotPrefix, isDisabled, prompt, textareaRef]
  );

  return {
    onChange,
    isOpen: assistMode !== null,
    onClose,
    onOpen,
    onSelect,
    onKeyDown,
    onFocus,
    cta: {
      isOpen: assistMode === 'cta',
      modelId,
      onSelect: onCtaSelect,
    },
  };
};
