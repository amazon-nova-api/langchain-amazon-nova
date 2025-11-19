# Testing Guidelines

## Unit Tests and Serialization

The unit tests in this project include serialization snapshot tests that are sensitive to Amazon Nova environment variables. To ensure consistent behavior between local development and CI environments, the following approach is used:

### AWS Credentials in Tests

- **CI Environment**: Does not have Amazon Nova credentials set as environment variables for unit tests
- **Local Environment**: May have Amazon Nova credentials set (eg. `NOVA_BEARER_TOKEN` or `NOVA_API_KEY`)
- **Test Behavior**: AWS credentials should NOT be included in serialization snapshots

### Running Tests

Always use the Makefile to run tests, which automatically excludes Amazon Nova environment variables:

```bash
# Run all unit tests
make test

# Run specific test file
make test TEST_FILE=tests/unit_tests/test_example.py

# Run specific test case
make test TEST_FILE=tests/unit_tests/test_example.py::TestClass::test_method
```

The Makefile automatically runs tests with these environment variables unset:

### Updating Snapshots

**⚠️ IMPORTANT**: When updating snapshots, always use the Makefile to ensure consistent results:

```bash
# Update snapshots for specific test
make test TEST_FILE=tests/unit_tests/test_standard.py --snapshot-update

# Never run this directly (will include Amazon Nova credentials):
# pytest tests/unit_tests/test_standard.py --snapshot-update
```
