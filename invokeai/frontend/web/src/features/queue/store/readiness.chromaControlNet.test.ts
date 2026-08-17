import { describe, expect, it, vi } from 'vitest';

vi.mock('features/dynamicPrompts/util/getShouldProcessPrompt', () => ({
  getShouldProcessPrompt: vi.fn(() => false),
}));

vi.mock('i18next', () => ({
  default: {
    t: (key: string) => key,
  },
}));

import type { CanvasState, ParamsState, RefImagesState } from 'features/controlLayers/store/types';
import type { DynamicPromptsState } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { MainModelConfig } from 'services/api/types';

import { getReasonsWhyCannotEnqueueCanvasTab } from './readiness';

const chromaModel = {
  key: 'chroma',
  hash: 'chroma-hash',
  name: 'Chroma1-HD',
  base: 'chroma',
  type: 'main',
  format: 'diffusers',
  submodels: {
    transformer: {},
    vae: {},
    text_encoder: {},
    tokenizer: {},
  },
} as unknown as MainModelConfig;

const dynamicPrompts = {
  prompts: ['test prompt'],
} as unknown as DynamicPromptsState;

const makeControlLayer = (isEnabled = true) => ({
  id: 'control-layer',
  isEnabled,
  objects: [{}],
  controlAdapter: {
    type: 'controlnet',
    model: {
      key: 'instantx-union',
      hash: 'instantx-union-hash',
      name: 'FLUX.1-dev-Controlnet-Union',
      base: 'flux',
      type: 'controlnet',
    },
    weight: 0.35,
    beginEndStepPct: [0, 0.3],
    controlMode: 'balanced',
    fluxControlType: { key: 'canny', instantxControlMode: 0 },
  },
});

const makeRedux = () => ({
  id: 'redux',
  isEnabled: true,
  config: {
    type: 'flux_redux',
    model: { key: 'redux-model', name: 'FLUX Redux', base: 'flux', type: 'flux_redux' },
    image: {
      original: { image: { image_name: 'reference.png', width: 1024, height: 1024 } },
    },
    downsamplingFactor: 2,
    weight: 1,
  },
});

const buildArg = ({
  scheduler = 'euler',
  controlEnabled = true,
  withRedux = false,
}: {
  scheduler?: string;
  controlEnabled?: boolean;
  withRedux?: boolean;
} = {}) => ({
  isConnected: true,
  model: chromaModel,
  canvas: {
    bbox: {
      scaleMethod: 'none',
      rect: { width: 1024, height: 1024 },
      scaledSize: { width: 1024, height: 1024 },
    },
    controlLayers: { entities: [makeControlLayer(controlEnabled)] },
    regionalGuidance: { entities: [] },
    rasterLayers: { entities: [] },
    inpaintMasks: { entities: [] },
  } as unknown as CanvasState,
  params: {
    positivePrompt: 'test',
    chromaScheduler: scheduler,
  } as unknown as ParamsState,
  refImages: {
    entities: withRedux ? [makeRedux()] : [],
  } as unknown as RefImagesState,
  loras: [],
  dynamicPrompts,
  canvasIsFiltering: false,
  canvasIsTransforming: false,
  canvasIsRasterizing: false,
  canvasIsCompositing: false,
  canvasIsSelectingObject: false,
  hasFlux2DiffusersVaeSource: false,
  hasFlux2DiffusersQwen3Source: false,
  hasFlux2DevDiffusersSource: false,
  wanWiredConfigs: { vae: null, componentSource: null, lowNoisePartner: null },
});

const hasReason = (reasons: { content: string }[], key: string) => reasons.some((reason) => reason.content === key);

describe('Chroma Canvas ControlNet readiness', () => {
  it('accepts both supported Chroma ControlNet schedulers', () => {
    for (const scheduler of ['euler', 'euler_cfg_pp_beta']) {
      const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildArg({ scheduler }));
      expect(hasReason(reasons, 'parameters.invoke.chromaControlNetUnsupportedScheduler')).toBe(false);
    }
  });

  it('blocks unsupported schedulers while an enabled Chroma ControlNet is present', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildArg({ scheduler: 'heun' }));
    expect(hasReason(reasons, 'parameters.invoke.chromaControlNetUnsupportedScheduler')).toBe(true);
  });

  it('blocks FLUX Redux together with an enabled Chroma ControlNet', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(buildArg({ withRedux: true }));
    expect(hasReason(reasons, 'parameters.invoke.chromaControlNetIncompatibleWithRedux')).toBe(true);
  });

  it('does not apply Chroma ControlNet combination rules to a disabled control layer', () => {
    const reasons = getReasonsWhyCannotEnqueueCanvasTab(
      buildArg({ scheduler: 'heun', controlEnabled: false, withRedux: true })
    );
    expect(hasReason(reasons, 'parameters.invoke.chromaControlNetUnsupportedScheduler')).toBe(false);
    expect(hasReason(reasons, 'parameters.invoke.chromaControlNetIncompatibleWithRedux')).toBe(false);
  });
});
