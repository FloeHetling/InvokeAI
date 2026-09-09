import type { FluxControlNetControlType } from 'features/controlLayers/store/types';

export type FluxControlNetControlTypeOption = FluxControlNetControlType & {
  label: string;
};

type FluxControlNetCapability = {
  description: string;
  controlTypes: readonly FluxControlNetControlTypeOption[];
};

const INSTANTX_UNION_CONTROL_TYPES = [
  { key: 'canny', label: 'Canny', instantxControlMode: 0 },
  { key: 'tile', label: 'Tile', instantxControlMode: 1 },
  { key: 'depth', label: 'Depth', instantxControlMode: 2 },
  { key: 'blur', label: 'Blur', instantxControlMode: 3 },
  { key: 'pose', label: 'Pose', instantxControlMode: 4 },
  { key: 'gray', label: 'Gray', instantxControlMode: 5 },
  { key: 'low_quality', label: 'Low Quality', instantxControlMode: 6 },
] as const satisfies readonly FluxControlNetControlTypeOption[];

const SHAKKER_UNION_PRO_2_CONTROL_TYPES = [
  { key: 'canny', label: 'Canny', instantxControlMode: null },
  { key: 'soft_edge', label: 'Soft Edge', instantxControlMode: null },
  { key: 'depth', label: 'Depth', instantxControlMode: null },
  { key: 'pose', label: 'Pose', instantxControlMode: null },
  { key: 'gray', label: 'Gray', instantxControlMode: null },
] as const satisfies readonly FluxControlNetControlTypeOption[];

const CAPABILITIES_BY_SOURCE = new Map<string, FluxControlNetCapability>([
  [
    'InstantX/FLUX.1-dev-Controlnet-Union',
    {
      description: 'Selects the explicit InstantX Union mode sent to the ControlNet.',
      controlTypes: INSTANTX_UNION_CONTROL_TYPES,
    },
  ],
  [
    'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
    {
      description:
        'Union Pro 2.0 has no mode embedding. This records the intended control image type; the model infers it from the image.',
      controlTypes: SHAKKER_UNION_PRO_2_CONTROL_TYPES,
    },
  ],
]);

type FluxControlNetModelLike = {
  base: string;
  type: string;
  source?: string | null;
};

export const getFluxControlNetCapability = (
  model: FluxControlNetModelLike | null | undefined
): FluxControlNetCapability | null => {
  if (!model || model.base !== 'flux' || model.type !== 'controlnet' || !model.source) {
    return null;
  }
  return CAPABILITIES_BY_SOURCE.get(model.source) ?? null;
};

export const getDefaultFluxControlNetControlType = (
  model: FluxControlNetModelLike | null | undefined
): FluxControlNetControlType | null => {
  const controlType = getFluxControlNetCapability(model)?.controlTypes[0];
  if (!controlType) {
    return null;
  }
  return {
    key: controlType.key,
    instantxControlMode: controlType.instantxControlMode,
  };
};

export const getInstantXControlModeForGraph = (controlType: FluxControlNetControlType | null): number =>
  controlType?.instantxControlMode ?? -1;

type ControlAdapterModelLike = {
  base: string;
  type: string;
};

/**
 * Canvas normally requires the control adapter and main model bases to match.
 * Chroma is the deliberate exception: its compatibility path consumes FLUX
 * InstantX ControlNet fields and residuals.
 */
export const isControlAdapterModelCompatible = (
  mainBase: string | null | undefined,
  adapterModel: ControlAdapterModelLike | null | undefined
): boolean => {
  if (!mainBase || !adapterModel) {
    return false;
  }
  if (mainBase === adapterModel.base) {
    return true;
  }
  return mainBase === 'chroma' && adapterModel.base === 'flux' && adapterModel.type === 'controlnet';
};
