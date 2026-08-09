import type { PropsWithChildren } from 'react';

export type PromptTriggerSelectProps = {
  onSelect: (v: string) => void;
  onClose: () => void;
};

type PromptPopoverCtaProps = {
  isOpen: boolean;
  modelId: string | null;
  onSelect: (renderedContent: string) => void;
};

export type PromptPopoverProps = PropsWithChildren &
  PromptTriggerSelectProps & {
    isOpen: boolean;
    width?: number | string;
    cta?: PromptPopoverCtaProps;
  };
