import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel, Tooltip } from '@invoke-ai/ui-library';
import {
  type FluxControlNetControlTypeOption,
  getFluxControlNetCapability,
} from 'features/controlLayers/store/fluxControlNet';
import type { FluxControlNetControlType } from 'features/controlLayers/store/types';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useControlLayerModels } from 'services/api/hooks/modelsByType';
import { assert } from 'tsafe';

type Props = {
  modelKey: string | null;
  controlType: FluxControlNetControlType | null;
  onChange: (controlType: FluxControlNetControlType) => void;
};

const toStateValue = (option: FluxControlNetControlTypeOption): FluxControlNetControlType => ({
  key: option.key,
  instantxControlMode: option.instantxControlMode,
});

export const ControlLayerFluxControlType = memo(({ modelKey, controlType, onChange }: Props) => {
  const [modelConfigs] = useControlLayerModels();
  const modelConfig = useMemo(() => modelConfigs.find((model) => model.key === modelKey), [modelConfigs, modelKey]);
  const capability = useMemo(() => getFluxControlNetCapability(modelConfig), [modelConfig]);

  const selectedControlType = useMemo(() => {
    if (!controlType) {
      return null;
    }
    return (
      capability?.controlTypes.find(
        (option) => option.key === controlType.key && option.instantxControlMode === controlType.instantxControlMode
      ) ?? null
    );
  }, [capability, controlType]);

  // Older persisted Canvas state predates fluxControlType. When such a layer is opened,
  // migrate it to the model's declared default instead of displaying Canny while still
  // sending the legacy -1 sentinel to a mode-conditioned Union model.
  useEffect(() => {
    if (!capability || selectedControlType) {
      return;
    }
    const defaultControlType = capability.controlTypes[0];
    if (defaultControlType) {
      onChange(toStateValue(defaultControlType));
    }
  }, [capability, onChange, selectedControlType]);

  const options = useMemo(
    () => capability?.controlTypes.map((option) => ({ label: option.label, value: option.key })) ?? [],
    [capability]
  );
  const value = useMemo(() => {
    const effectiveControlType = selectedControlType ?? capability?.controlTypes[0];
    return options.find((option) => option.value === effectiveControlType?.key);
  }, [capability, options, selectedControlType]);

  const handleChange = useCallback<ComboboxOnChange>(
    (option) => {
      const selected = capability?.controlTypes.find((controlType) => controlType.key === option?.value);
      assert(selected);
      onChange(toStateValue(selected));
    },
    [capability, onChange]
  );

  if (!capability) {
    return null;
  }

  return (
    <FormControl>
      <Tooltip label={capability.description}>
        <FormLabel m={0}>Control Type</FormLabel>
      </Tooltip>
      <Combobox value={value} options={options} onChange={handleChange} isClearable={false} isSearchable={false} />
    </FormControl>
  );
});

ControlLayerFluxControlType.displayName = 'ControlLayerFluxControlType';
