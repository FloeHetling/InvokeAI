import type { ClipTagType } from '@features/cliptags/core/types';

import { Badge } from '@chakra-ui/react';
import { CLIP_TAG_TYPE_PALETTE, CLIP_TAG_TYPES } from '@features/cliptags/core/types';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { Option } from './OptionSelect';

export const isClipTagType = (value: string): value is ClipTagType =>
  (CLIP_TAG_TYPES as readonly string[]).includes(value);

export const TagTypeBadge = ({ type }: { type: ClipTagType }) => {
  const { t } = useTranslation();

  return (
    <Badge colorPalette={CLIP_TAG_TYPE_PALETTE[type]} fontSize="xs" variant="surface">
      {t(`cliptags.types.${type}`)}
    </Badge>
  );
};

/** Every tag type as a select option, in the backend's order. */
export const useTagTypeOptions = (): Option[] => {
  const { t } = useTranslation();

  return useMemo(() => CLIP_TAG_TYPES.map((type) => ({ label: t(`cliptags.types.${type}`), value: type })), [t]);
};
