import type { ControlAdapterKind } from '@features/generation/graph';
import type { ModelConfig } from '@features/models';

import { isAdapterBaseCompatible } from '@features/generation/graph';

export const getCompatibleControlModels = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): ModelConfig[] => {
  if (kind === 'z_image_control' && base !== 'z-image') {
    return [];
  }
  const modelType = kind === 'z_image_control' ? 'controlnet' : kind;
  return models.filter((model) => model.type === modelType && (!base || isAdapterBaseCompatible(base, model.base)));
};

/**
 * The control model a freshly created layer should start with: a union model
 * (the generalist; the lighter "Pro" build of it where both are installed), then
 * tile, then the first compatible model. Null when no compatible model is
 * installed.
 */
export const resolveDefaultControlModel = (
  models: readonly ModelConfig[],
  base: string | null,
  kind: ControlAdapterKind
): string | null => {
  const compatible = getCompatibleControlModels(models, base, kind);
  const unions = compatible.filter((model) => model.name.toLowerCase().includes('union'));
  // FLUX and Chroma ship a Union Pro build that is the lighter, better-behaved one; other bases keep the first union.
  const prefersPro = base === 'flux' || base === 'chroma';
  const preferred =
    (prefersPro ? unions.find((model) => /\bpro\b/i.test(model.name)) : undefined) ??
    unions[0] ??
    compatible.find((model) => model.name.toLowerCase().includes('tile')) ??
    compatible[0];
  return preferred?.key ?? null;
};

/** Default model for the adapter kind the layer factories pick for `base`. */
export const resolveDefaultControlModelForBase = (models: readonly ModelConfig[], base: string | null): string | null =>
  resolveDefaultControlModel(models, base, base === 'z-image' ? 'z_image_control' : 'controlnet');
