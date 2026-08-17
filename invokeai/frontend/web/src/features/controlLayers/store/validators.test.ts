import {
  getControlLayerState,
  getReferenceImageState,
  getRegionalGuidanceState,
  initialRegionalGuidanceIPAdapter,
} from 'features/controlLayers/store/util';
import {
  getControlLayerWarnings,
  getGlobalReferenceImageWarnings,
  getRegionalGuidanceWarnings,
} from 'features/controlLayers/store/validators';
import { describe, expect, it } from 'vitest';

const krea2Model = { base: 'krea-2' } as never;
const chromaModel = { base: 'chroma' } as never;

describe('getRegionalGuidanceWarnings - Krea-2', () => {
  it('allows positive regional prompts', () => {
    const region = getRegionalGuidanceState('region', { positivePrompt: 'red fox' });

    const warnings = getRegionalGuidanceWarnings(region, krea2Model);

    expect(warnings).not.toContain('controlLayers.warnings.rgNegativePromptNotSupported');
    expect(warnings).not.toContain('controlLayers.warnings.rgAutoNegativeNotSupported');
    expect(warnings).not.toContain('controlLayers.warnings.rgReferenceImagesNotSupported');
  });

  it('warns for unsupported negative prompts and auto-negative', () => {
    const region = getRegionalGuidanceState('region', {
      positivePrompt: 'red fox',
      negativePrompt: 'blue fox',
      autoNegative: true,
    });

    const warnings = getRegionalGuidanceWarnings(region, krea2Model);

    expect(warnings).toContain('controlLayers.warnings.rgNegativePromptNotSupported');
    expect(warnings).toContain('controlLayers.warnings.rgAutoNegativeNotSupported');
  });

  it('warns for unsupported regional reference images', () => {
    const region = getRegionalGuidanceState('region', {
      positivePrompt: 'red fox',
      referenceImages: [{ id: 'reference', config: initialRegionalGuidanceIPAdapter }],
    });

    const warnings = getRegionalGuidanceWarnings(region, krea2Model);

    expect(warnings).toContain('controlLayers.warnings.rgReferenceImagesNotSupported');
  });
});

describe('Chroma unsupported adapters', () => {
  it('warns for global reference images', () => {
    const warnings = getGlobalReferenceImageWarnings(getReferenceImageState('reference'), chromaModel);

    expect(warnings).toContain('controlLayers.warnings.unsupportedModel');
  });

  it('warns for control layers even when their adapter base matches', () => {
    const layer = getControlLayerState('control', {
      controlAdapter: { model: { base: 'chroma' } as never },
    });

    const warnings = getControlLayerWarnings(layer, chromaModel);

    expect(warnings).toContain('controlLayers.warnings.unsupportedModel');
  });
});
