import { Popover, PopoverAnchor, PopoverBody, PopoverContent, Portal } from '@invoke-ai/ui-library';
import { ClipTagAutocompleteSelect } from 'features/clipTagAutocomplete/components/ClipTagAutocompleteSelect';
import { PromptTriggerSelect } from 'features/prompt/PromptTriggerSelect';
import type { PromptPopoverProps } from 'features/prompt/types';
import { memo } from 'react';

export const PromptPopover = memo((props: PromptPopoverProps) => {
  const { onSelect, isOpen, onClose, width, children, cta } = props;
  const popoverWidth =
    width === undefined ? undefined : `calc(${typeof width === 'number' ? `${width}px` : width} - 0.25rem)`;

  return (
    <Popover
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      // The CTA Combobox menu is portaled to document.body. If the Popover closes on blur,
      // a pointer click is treated as outside interaction and unmounts the menu before
      // react-select can deliver the option click. Keyboard selection is unaffected.
      closeOnBlur={!cta?.isOpen}
      returnFocusOnClose={false}
      isLazy
    >
      <PopoverAnchor>{children}</PopoverAnchor>
      <Portal>
        <PopoverContent
          p={0}
          insetBlockStart={-1}
          shadow="dark-lg"
          borderColor="invokeBlue.300"
          borderWidth="2px"
          borderStyle="solid"
        >
          <PopoverBody p={0} width={popoverWidth}>
            {cta?.isOpen ? (
              <ClipTagAutocompleteSelect modelId={cta.modelId} onClose={onClose} onSelect={cta.onSelect} />
            ) : (
              <PromptTriggerSelect onClose={onClose} onSelect={onSelect} />
            )}
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

PromptPopover.displayName = 'PromptPopover';
