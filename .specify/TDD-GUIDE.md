# Test-Driven Development (TDD) Guide

## Mandatory TDD Workflow

**Every single feature, component, or function MUST follow this exact process:**

### The Red-Green-Refactor Cycle

```
1. RED    → Write a failing test
2. GREEN  → Write minimal code to pass
3. REFACTOR → Improve code quality
4. REPEAT → Next behavior
```

## Step-by-Step Process

### Step 1: Write Test First (RED)

```bash
# Create test file
touch tests/test_<component>.py
```

```python
# tests/test_user_auth.py
def test_user_can_login_with_valid_credentials():
    """Test that a user can log in with correct username and password."""
    # Arrange
    user = User(username="test", password="password123")
    
    # Act
    result = authenticate(user.username, user.password)
    
    # Assert
    assert result.is_authenticated == True
    assert result.user_id == user.id
```

### Step 2: Run Test - Verify Failure (RED)

```bash
pytest tests/test_user_auth.py::test_user_can_login_with_valid_credentials -v
```

**Expected output:**
```
FAILED tests/test_user_auth.py::test_user_can_login_with_valid_credentials
ImportError: cannot import name 'authenticate'
```

✅ **Good!** Test fails as expected.

### Step 3: Write Minimal Implementation (GREEN)

```python
# src/auth.py
def authenticate(username: str, password: str):
    """Authenticate user with username and password."""
    # Minimal implementation - just enough to pass
    user = get_user_by_username(username)
    if user and user.password == password:
        return AuthResult(is_authenticated=True, user_id=user.id)
    return AuthResult(is_authenticated=False, user_id=None)
```

### Step 4: Run Test - Verify Success (GREEN)

```bash
pytest tests/test_user_auth.py::test_user_can_login_with_valid_credentials -v
```

**Expected output:**
```
PASSED tests/test_user_auth.py::test_user_can_login_with_valid_credentials
```

✅ **Perfect!** Test passes.

### Step 5: Commit (Paired Commits)

```bash
# First commit: the test
git add tests/test_user_auth.py
git commit -m "test: add test for user authentication with valid credentials"

# Second commit: the implementation
git add src/auth.py
git commit -m "feat: implement basic user authentication"
```

### Step 6: Refactor (Optional)

```python
# src/auth.py - improved version
def authenticate(username: str, password: str) -> AuthResult:
    """
    Authenticate user with username and password.
    
    Args:
        username: User's username
        password: User's password (plaintext)
        
    Returns:
        AuthResult with authentication status and user ID
    """
    user = get_user_by_username(username)
    
    if not user:
        return AuthResult(is_authenticated=False, user_id=None)
        
    if not verify_password(user.password_hash, password):
        return AuthResult(is_authenticated=False, user_id=None)
        
    return AuthResult(is_authenticated=True, user_id=user.id)
```

Run tests again to ensure refactoring didn't break anything:

```bash
pytest tests/test_user_auth.py -v
```

Commit refactoring:

```bash
git add src/auth.py
git commit -m "refactor: improve authentication code clarity and security"
```

### Step 7: Next Behavior - Repeat

Now write the NEXT test for the NEXT behavior:

```python
# tests/test_user_auth.py
def test_user_cannot_login_with_invalid_password():
    """Test that authentication fails with wrong password."""
    user = User(username="test", password="password123")
    
    result = authenticate(user.username, "wrong_password")
    
    assert result.is_authenticated == False
    assert result.user_id is None
```

Start the cycle again: RED → GREEN → REFACTOR → REPEAT

## Granularity Rules

### ✅ Good Granularity (Tiny Steps)

```python
# Test 1: User can be created
def test_create_user():
    user = create_user("john", "john@example.com")
    assert user.username == "john"

# Test 2: User email must be valid
def test_create_user_validates_email():
    with pytest.raises(ValueError):
        create_user("john", "invalid-email")

# Test 3: Username must be unique
def test_create_user_enforces_unique_username():
    create_user("john", "john@example.com")
    with pytest.raises(ValueError):
        create_user("john", "john2@example.com")
```

Each test verifies ONE specific behavior.

### ❌ Bad Granularity (Too Big)

```python
# DON'T DO THIS - Testing too much at once
def test_entire_user_registration_flow():
    # Creates user
    user = create_user("john", "john@example.com")
    
    # Validates email
    assert validate_email(user.email)
    
    # Sends welcome email
    assert email_sent_to(user.email)
    
    # Logs user in
    session = login(user.username, "password")
    assert session.is_active
```

This test is too coarse-grained. Break it into 4+ separate tests.

## Component-Level TDD

### Example: Building a User Service

**Feature**: User Registration

**Components to test separately:**
1. User model
2. Email validator
3. Password hasher
4. User repository
5. Registration service
6. Registration endpoint

**TDD Order:**

```
1. Test user model → Implement user model
2. Test email validator → Implement email validator
3. Test password hasher → Implement password hasher
4. Test user repository → Implement user repository
5. Test registration service → Implement registration service
6. Test registration endpoint → Implement registration endpoint
7. Integration test → Wire components together
```

## Integration Testing (After Unit Tests)

Only test integration AFTER all unit tests pass:

```python
# tests/integration/test_user_registration.py
def test_user_registration_end_to_end():
    """Integration test: complete registration flow."""
    # This test only runs after all unit tests pass
    response = client.post("/api/register", json={
        "username": "john",
        "email": "john@example.com",
        "password": "secure123"
    })
    
    assert response.status_code == 201
    assert response.json()["username"] == "john"
    
    # Verify user was actually created
    user = get_user_by_username("john")
    assert user is not None
    assert user.email == "john@example.com"
```

## Common Patterns

### Pattern 1: Model Testing

```python
# Test first
def test_user_model_stores_username():
    user = User(username="john")
    assert user.username == "john"

# Implement
class User:
    def __init__(self, username: str):
        self.username = username
```

### Pattern 2: Service Testing

```python
# Test first
def test_user_service_creates_user():
    service = UserService()
    user = service.create_user("john", "john@example.com")
    assert user.username == "john"

# Implement
class UserService:
    def create_user(self, username: str, email: str) -> User:
        return User(username=username, email=email)
```

### Pattern 3: API Endpoint Testing

```python
# Test first
def test_register_endpoint_returns_201():
    response = client.post("/api/register", json={
        "username": "john",
        "email": "john@example.com"
    })
    assert response.status_code == 201

# Implement
@app.post("/api/register")
def register(data: RegisterRequest):
    user = user_service.create_user(data.username, data.email)
    return {"id": user.id}, 201
```

## Commit Pattern

### Good Commit History (TDD)

```
test: add test for user creation
feat: implement user creation
test: add test for email validation
feat: implement email validation
test: add test for password hashing
feat: implement password hashing
refactor: extract validation logic
test: add integration test for registration
feat: wire up registration components
```

### Bad Commit History (Not TDD)

```
feat: implement entire user registration system
test: add tests for user registration
fix: bugs found during testing
```

## Quality Checklist

Before merging any PR, verify:

- [ ] Every implementation commit has a preceding test commit
- [ ] All tests were written BEFORE implementation
- [ ] Each test verifies ONE specific behavior
- [ ] Tests are fine-grained (function/method level)
- [ ] Integration tests only after unit tests pass
- [ ] No big bang commits
- [ ] All tests passing (green)
- [ ] 100% coverage for new code

## Anti-Patterns to Avoid

### ❌ Writing Tests After Implementation

```bash
# WRONG ORDER
git commit -m "feat: implement user authentication"
git commit -m "test: add tests for authentication"
```

### ❌ Testing Multiple Behaviors at Once

```python
# DON'T DO THIS
def test_user_authentication_and_authorization():
    # Too much in one test
    pass
```

### ❌ Skipping Tests for "Simple" Code

```python
# WRONG - Even simple code needs tests
def add(a, b):
    return a + b  # "Too simple to test" - NO!
```

### ❌ Big Bang Implementation

```python
# WRONG - Implementing entire feature at once
def complete_user_registration_system():
    # 500 lines of code without tests
    pass
```

## Remember

1. **Test first, always**
2. **One test at a time**
3. **Smallest possible increment**
4. **Run tests after every change**
5. **Keep tests green**
6. **Commit test before implementation**
7. **Integration only after units pass**
8. **No big bang - tiny steps only**

---

**TDD is not optional. It's the foundation of quality software.**
