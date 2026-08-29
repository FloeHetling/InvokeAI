import {
  getControlLayerState,
  getReferenceImageState,
  getRegionalGuidanceState,
  initialFLUXRedux,
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
  it('allows FLUX Redux global reference images', () => {
    const warnings = getGlobalReferenceImageWarnings(
      getReferenceImageState('reference', {
        config: {
          ...initialFLUXRedux,
          image: {} as never,
          model: {
            base: 'flux',
            key: 'flux-redux',
            name: 'FLUX Redux',
            type: 'flux_redux',
          } as never,
        },
      }),
      chromaModel
    );

    expect(warnings).not.toContain('controlLayers.warnings.unsupportedModel');
    expect(warnings).not.toContain('controlLayers.warnings.ipAdapterIncompatibleBaseModel');
  });

  it('warns when a Chroma Redux config references a non-Redux model', () => {
    const warnings = getGlobalReferenceImageWarnings(
      getReferenceImageState('reference', {
        config: {
          ...initialFLUXRedux,
          image: {} as never,
          model: {
            base: 'flux',
            key: 'flux-ip-adapter',
            name: 'FLUX IP Adapter',
            type: 'ip_adapter',
          } as never,
        },
      }),
      chromaModel
    );

    expect(warnings).toContain('controlLayers.warnings.ipAdapterIncompatibleBaseModel');
  });

  it('warns for IP-adapter global reference images', () => {
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
