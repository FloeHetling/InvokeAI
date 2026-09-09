import { describe, expect, it } from 'vitest';

import {
  getDefaultFluxControlNetControlType,
  getFluxControlNetCapability,
  getInstantXControlModeForGraph,
  isControlAdapterModelCompatible,
} from './fluxControlNet';

describe('FLUX ControlNet capabilities', () => {
  it('maps the documented InstantX Union modes to their explicit mode indices', () => {
    const capability = getFluxControlNetCapability({
      base: 'flux',
      type: 'controlnet',
      source: 'InstantX/FLUX.1-dev-Controlnet-Union',
    });

    expect(capability?.controlTypes).toEqual([
      { key: 'canny', label: 'Canny', instantxControlMode: 0 },
      { key: 'tile', label: 'Tile', instantxControlMode: 1 },
      { key: 'depth', label: 'Depth', instantxControlMode: 2 },
      { key: 'blur', label: 'Blur', instantxControlMode: 3 },
      { key: 'pose', label: 'Pose', instantxControlMode: 4 },
      { key: 'gray', label: 'Gray', instantxControlMode: 5 },
      { key: 'low_quality', label: 'Low Quality', instantxControlMode: 6 },
    ]);
    expect(
      getDefaultFluxControlNetControlType({
        base: 'flux',
        type: 'controlnet',
        source: 'InstantX/FLUX.1-dev-Controlnet-Union',
      })
    ).toEqual({ key: 'canny', instantxControlMode: 0 });
  });

  it('keeps Shakker Union Pro 2.0 control types semantic because the checkpoint has no mode embedding', () => {
    const capability = getFluxControlNetCapability({
      base: 'flux',
      type: 'controlnet',
      source: 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
    });

    expect(capability?.controlTypes).toEqual([
      { key: 'canny', label: 'Canny', instantxControlMode: null },
      { key: 'soft_edge', label: 'Soft Edge', instantxControlMode: null },
      { key: 'depth', label: 'Depth', instantxControlMode: null },
      { key: 'pose', label: 'Pose', instantxControlMode: null },
      { key: 'gray', label: 'Gray', instantxControlMode: null },
    ]);
    expect(
      getDefaultFluxControlNetControlType({
        base: 'flux',
        type: 'controlnet',
        source: 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
      })
    ).toEqual({ key: 'canny', instantxControlMode: null });
  });

  it('uses the backend -1 sentinel when the selected model has no explicit mode embedding', () => {
    expect(getInstantXControlModeForGraph({ key: 'canny', instantxControlMode: 0 })).toBe(0);
    expect(getInstantXControlModeForGraph({ key: 'canny', instantxControlMode: null })).toBe(-1);
    expect(getInstantXControlModeForGraph(null)).toBe(-1);
  });

  it('does not invent semantic modes for unknown FLUX ControlNets', () => {
    expect(
      getFluxControlNetCapability({
        base: 'flux',
        type: 'controlnet',
        source: 'Example/Unknown-ControlNet',
      })
    ).toBeNull();
  });

  it('allows FLUX ControlNets on Chroma without opening other cross-base adapters', () => {
    expect(isControlAdapterModelCompatible('chroma', { base: 'flux', type: 'controlnet' })).toBe(true);
    expect(isControlAdapterModelCompatible('chroma', { base: 'flux', type: 'control_lora' })).toBe(false);
    expect(isControlAdapterModelCompatible('chroma', { base: 'sdxl', type: 'controlnet' })).toBe(false);
    expect(isControlAdapterModelCompatible('flux', { base: 'flux', type: 'controlnet' })).toBe(true);
  });
});
