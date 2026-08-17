import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const DETAIL_MIN = 1;
const DETAIL_MAX = 9;
const DETAIL_DEFAULT = 2;

const STRENGTH_MIN = 0;
const STRENGTH_MAX = 1;
const STRENGTH_DEFAULT = 1;
const STRENGTH_MARKS = [0, 0.25, 0.5, 0.75, 1];

const formatStrength = (value: number) => value.toFixed(2);

type Props = {
  downsamplingFactor: number;
  weight: number;
  onChangeDownsamplingFactor: (downsamplingFactor: number) => void;
  onChangeWeight: (weight: number) => void;
};

export const FLUXReduxGlobalSettings = memo(
  ({ downsamplingFactor, weight, onChangeDownsamplingFactor, onChangeWeight }: Props) => {
    const { t } = useTranslation();

    return (
      <Flex flexDir="column" gap={2} w="full">
        <FormControl>
          <Flex alignItems="center" gap={4} w="full">
            <FormLabel m={0}>{t('controlLayers.fluxReduxSettings.detail')}</FormLabel>
            <Flex flexDir="column" gap={1} w="full">
              <CompositeSlider
                min={DETAIL_MIN}
                max={DETAIL_MAX}
                step={1}
                fineStep={1}
                value={downsamplingFactor}
                defaultValue={DETAIL_DEFAULT}
                onChange={onChangeDownsamplingFactor}
              />
              <Flex justifyContent="space-between" px={1}>
                <Text color="base.500" fontSize="xs">
                  {t('controlLayers.fluxReduxSettings.moreDetail')}
                </Text>
                <Text color="base.500" fontSize="xs">
                  {t('controlLayers.fluxReduxSettings.lessDetail')}
                </Text>
              </Flex>
            </Flex>
            <CompositeNumberInput
              maxW={20}
              min={DETAIL_MIN}
              max={DETAIL_MAX}
              step={1}
              fineStep={1}
              value={downsamplingFactor}
              defaultValue={DETAIL_DEFAULT}
              onChange={onChangeDownsamplingFactor}
            />
          </Flex>
        </FormControl>
        <FormControl orientation="horizontal">
          <FormLabel m={0}>{t('controlLayers.fluxReduxSettings.strength')}</FormLabel>
          <CompositeSlider
            min={STRENGTH_MIN}
            max={STRENGTH_MAX}
            step={0.05}
            fineStep={0.01}
            value={weight}
            defaultValue={STRENGTH_DEFAULT}
            onChange={onChangeWeight}
            marks={STRENGTH_MARKS}
            formatValue={formatStrength}
          />
          <CompositeNumberInput
            maxW={20}
            min={STRENGTH_MIN}
            max={STRENGTH_MAX}
            step={0.05}
            fineStep={0.01}
            value={weight}
            defaultValue={STRENGTH_DEFAULT}
            onChange={onChangeWeight}
          />
        </FormControl>
      </Flex>
    );
  }
);

FLUXReduxGlobalSettings.displayName = 'FLUXReduxGlobalSettings';
