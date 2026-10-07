import type { SelectRootProps, SelectValueChangeDetails } from '@chakra-ui/react';

import { createListCollection } from '@chakra-ui/react';
import { Select } from '@platform/ui/Select';
import { useCallback, useMemo } from 'react';

export interface Option {
  label: string;
  value: string;
}

export interface OptionSelectProps extends Omit<
  SelectRootProps<Option>,
  'children' | 'collection' | 'multiple' | 'onChange' | 'onValueChange' | 'value'
> {
  /** Names the control for assistive technology; inside a Field the field's label names it instead. */
  label: string;
  options: readonly Option[];
  /** Shown while `value` matches no option. */
  placeholder?: string;
  /** Inside a dialog the menu stays in the dialog's focus scope instead of a portal. */
  inDialog?: boolean;
  value: string;
  onChange: (value: string) => void;
}

/** A single-choice Select over plain `{ value, label }` options. */
export const OptionSelect = ({
  inDialog = false,
  label,
  onChange,
  options,
  placeholder,
  value,
  ...rest
}: OptionSelectProps) => {
  const collection = useMemo(() => createListCollection({ items: [...options] }), [options]);
  const selected = useMemo(() => [value], [value]);
  const valueTextProps = useMemo(() => ({ placeholder }), [placeholder]);
  const handleValueChange = useCallback(
    (details: SelectValueChangeDetails<Option>) => {
      const next = details.value[0];

      if (next !== undefined) {
        onChange(next);
      }
    },
    [onChange]
  );

  return (
    <Select
      aria-label={label}
      collection={collection}
      portalled={!inDialog}
      value={selected}
      valueTextProps={valueTextProps}
      onValueChange={handleValueChange}
      {...rest}
    />
  );
};
