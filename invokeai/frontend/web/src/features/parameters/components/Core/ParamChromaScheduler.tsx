import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectChromaScheduler, setChromaScheduler } from 'features/controlLayers/store/paramsSlice';
import { isParameterChromaScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const CHROMA_SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'euler_cfg_pp_beta', label: 'Euler CFG++ (Beta)' },
  { value: 'heun', label: 'Heun (2nd order)' },
  { value: 'lcm', label: 'LCM' },
];

const ParamChromaScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scheduler = useAppSelector(selectChromaScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterChromaScheduler(v?.value)) {
        return;
      }
      dispatch(setChromaScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => CHROMA_SCHEDULER_OPTIONS.find((o) => o.value === scheduler), [scheduler]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={CHROMA_SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamChromaScheduler);
