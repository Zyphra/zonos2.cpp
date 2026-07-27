const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const uiPath = path.join(__dirname, '..', 'web', 'tts_ui.html');
const uiSource = fs.readFileSync(uiPath, 'utf8');

function extractFunction(name) {
  const pattern = new RegExp(`    function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n    \\}`);
  const match = uiSource.match(pattern);
  assert.ok(match, `could not find ${name} in ${uiPath}`);
  return match[0];
}

class FakeSelect {
  constructor(optionIds) {
    this.optionIds = new Set(optionIds);
    this.selectedValue = '';
  }

  setOptions(optionIds) {
    this.optionIds = new Set(optionIds);
    if (!this.optionIds.has(this.selectedValue)) this.selectedValue = '';
  }

  get value() {
    return this.selectedValue;
  }

  set value(nextValue) {
    // HTMLSelectElement.value clears the selection when no option matches.
    this.selectedValue = this.optionIds.has(nextValue) ? nextValue : '';
  }
}

test('an uploaded speaker is selected only after its option is rendered', () => {
  const applyUploadedSpeakers = new Function(
    'renderSpeakerSelectOptions',
    'autoAssignSpeakers',
    'renderSpeakerLibrary',
    'updateSpeakerBlendUI',
    `${extractFunction('applyUploadedSpeakers')}; return applyUploadedSpeakers;`,
  )(
    renderSpeakerSelectOptions,
    autoAssignSpeakers,
    () => calls.push('library'),
    () => calls.push('blend'),
  );

  const uploadedSpeaker = { id: 'spk_uploaded' };
  const speakerASelect = new FakeSelect(['']);
  const calls = [];

  function renderSpeakerSelectOptions() {
    calls.push('options');
    speakerASelect.setOptions(['', uploadedSpeaker.id]);
  }

  function autoAssignSpeakers(addedSpeakers) {
    calls.push('assign');
    speakerASelect.value = addedSpeakers[addedSpeakers.length - 1].id;
  }

  applyUploadedSpeakers([uploadedSpeaker]);

  assert.equal(speakerASelect.value, uploadedSpeaker.id);
  assert.deepEqual(calls, ['options', 'assign', 'library', 'blend']);
  assert.match(uiSource, /applyUploadedSpeakers\(addedSpeakers\);/);
});
