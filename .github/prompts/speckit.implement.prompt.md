# SpecKit Implementation Prompt - TDD Enforced

When implementing features using `/speckit.implement`, follow this TDD workflow:

## Before Starting

1. Read `.specify/constitution.md` - TDD is PRIMARY PRINCIPLE
2. Read `.specify/TDD-GUIDE.md` - Detailed TDD workflow
3. Review `tasks.md` for current task

## Implementation Process (TDD Mandatory)

### For Each Task:

#### Step 1: Write Test First
```bash
# Create or update test file
# Write ONE test for ONE behavior
# Test MUST fail (RED)
```

#### Step 2: Run Test - Verify Failure
```bash
pytest tests/test_<component>.py -v
# Confirm test fails as expected
```

#### Step 3: Write Minimal Implementation
```bash
# Write ONLY enough code to pass the test
# No extra features
# No "nice-to-haves"
```

#### Step 4: Run Test - Verify Success
```bash
pytest tests/test_<component>.py -v
# Confirm test passes (GREEN)
```

#### Step 5: Commit (Paired Commits)
```bash
git add tests/
git commit -m "test: add test for <behavior>"

git add src/
git commit -m "feat: implement <behavior>"
```

#### Step 6: Refactor (Optional)
```bash
# Improve code quality
# Keep tests green
# Commit refactoring separately
```

#### Step 7: Repeat
```bash
# Next test, next behavior
# Tiny incremental steps
```

## Rules

### ✅ Always Do
- Write test BEFORE implementation
- One test at a time
- Smallest possible increment
- Run tests after every change
- Commit test before implementation
- Keep all tests passing

### ❌ Never Do
- Big bang implementations
- Multiple features at once
- Implementation before tests
- Skipping tests
- Committing untested code

## Quality Checks

Before marking task complete:
- [ ] Test written first (verify in git history)
- [ ] Test failed initially (RED)
- [ ] Implementation makes test pass (GREEN)
- [ ] Code refactored if needed
- [ ] All tests still passing
- [ ] Paired commits (test → implementation)
- [ ] 100% coverage for new code

## Integration Testing

Only after ALL unit tests pass:
1. Write integration test (RED)
2. Wire components together (GREEN)
3. Refactor if needed
4. Commit

## Remember

**TDD is not optional. Every line of code must have a test written FIRST.**

See `.specify/TDD-GUIDE.md` for detailed examples and patterns.
