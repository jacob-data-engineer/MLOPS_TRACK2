import rules from './trainingParameterRules.json';

type Rule = {kind: string; min?: number; max?: number; exclusive_min?: boolean; choices?: string[]};
const schema: Record<string, Rule> = rules;

export function trainingParameterError(key: string, value: unknown): string {
  const rule = schema[key];
  if (!rule || value === undefined || value === null || value === '') return '';
  const text = String(value).trim();
  if (!text) return '';
  if (rule.kind === 'freeze') {
    if (text.toLowerCase() === 'none') return '';
    const parts = text.replace(/^\[|\]$/g, '').split(',').map(p => p.trim());
    return parts.length && parts.every(p => /^\d+$/.test(p)) ? ''
      : 'Use a whole layer count or comma-separated layer indices. Minimum: 0; maximum: no fixed limit (depends on the model).';
  }
  if (rule.kind === 'bool' || rule.kind === 'amp') {
    const allowed = ['true','false','1','0','yes','no', ...(rule.kind === 'amp' ? ['fp16','bf16','fp32'] : [])];
    return allowed.includes(text.toLowerCase()) ? '' : `Allowed values: ${allowed.join(', ')}.`;
  }
  if (rule.kind === 'choice') return rule.choices?.some(c => c.toLowerCase() === text.toLowerCase()) ? '' : `Choose one of: ${rule.choices?.join(', ')}.`;
  const n = Number(text);
  const numeric = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/.test(text) && typeof value !== 'boolean';
  const invalid = !numeric || !Number.isFinite(n) || (rule.kind === 'int' && !Number.isInteger(n))
    || (rule.min !== undefined && (rule.exclusive_min ? n <= rule.min : n < rule.min))
    || (rule.max !== undefined && n > rule.max);
  if (!invalid) return '';
  return `${rule.kind === 'int' ? 'Enter a whole number.' : 'Enter a finite number.'} Minimum: ${rule.min === undefined ? 'no fixed limit' : `${rule.exclusive_min ? 'greater than ' : ''}${rule.min}`}; maximum: ${rule.max ?? 'no fixed limit'}.`;
}

export function trainingSettingsErrors(shape: {epochs:number; imgSize:number; batch:number; seed:number; patience:number; hyperparams:Record<string,unknown>}): string[] {
  return Object.entries({epochs:shape.epochs,img_size:shape.imgSize,batch:shape.batch,seed:shape.seed,patience:shape.patience,...shape.hyperparams})
    .flatMap(([key,value]) => {const error = trainingParameterError(key,value); return error ? [`${key}: ${error}`] : [];});
}
