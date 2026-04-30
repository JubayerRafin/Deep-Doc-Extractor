"""Validate Stage 1 output against mentor sample."""
import re

with open('output/stage1/hp-e877-series-user-guide_pages_46_47.md') as f:
    our = f.read()
with open('sample_35-36.md') as f:
    mentor = f.read()

print('=' * 70)
print('  VALIDATION: Our Output vs Mentor Sample')
print('=' * 70)

checks = []

# Structure checks
checks.append(('## Heading detected',
    '## Remove and replace the toner cartridge' in our))
checks.append(('Intro paragraph',
    'Follow these steps to replace the toner cartridge.' in our))
checks.append(('NOTE block (toner empty)',
    '**NOTE:** When a toner cartridge is empty' in our))
checks.append(('CAUTION block (force)',
    '**CAUTION:** Do not try to pull the toner cartridge' in our))

# All 7 steps
for i in range(1, 8):
    checks.append((f'Step {i} present',
        bool(re.search(rf'^{i}\.', our, re.MULTILINE))))

# Images
our_images = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', our)
mentor_images = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', mentor)
checks.append((f'Image count = 7', len(our_images) == 7))

# Alt text quality
alt_keywords = [
    (1, 'Open front door'),
    (2, 'Eject'),
    (3, 'Unpack new cartridge'),
    (4, 'Rock cartridge'),
    (5, 'Insert cartridge'),
    (6, 'Close front door'),
    (7, 'Pack used cartridge'),
]
for step_num, kw in alt_keywords:
    found = any(f'Step {step_num}' in alt and kw in alt for alt, _ in our_images)
    checks.append((f'  Step {step_num} alt text contains "{kw}"', found))

# No false positive tables
checks.append(('No false-positive tables', '```json' not in our))

# Image follows step in order
lines = our.split('\n')
for i in range(1, 8):
    step_idx = None
    img_found = False
    for idx, line in enumerate(lines):
        if re.match(rf'^{i}\.', line):
            step_idx = idx
        if step_idx is not None and idx > step_idx and line.startswith(f'![Step {i}:'):
            img_found = True
            break
    checks.append((f'  Step {i} image follows step text', img_found))

# Print results
passed = sum(1 for _, ok in checks if ok)
failed = sum(1 for _, ok in checks if not ok)
for label, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}  {label}')

print(f'\n  Results: {passed}/{passed + failed} checks passed')
if failed == 0:
    print('  >>> ALL CHECKS PASSED — output matches mentor format <<<')
else:
    print(f'  !!! {failed} check(s) failed')

print('\n-- Image Comparison --')
print(f'  Mentor: {len(mentor_images)} images')
for alt, path in mentor_images:
    print(f'    [{alt}]')
print(f'  Ours:   {len(our_images)} images')
for alt, path in our_images:
    print(f'    [{alt}]')
