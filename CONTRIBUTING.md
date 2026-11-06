# Contributing

Thanks for considering a contribution!

## Before opening a pull request

- Keep changes focused and explain the user or engineering problem they solve.
- Run the application locally and verify the affected workflow.
- Run the repository smoke checks where possible:
  ```bash
  python -m compileall -q backend api
  node --check motion-blur.js
  ```
- Update the README when changing setup instructions, API fields, or processing behavior.
- Never commit secrets, credentials, private images, generated outputs, or virtual environments.

## Pull request checklist

- [ ] The change has a clear description.
- [ ] Relevant local checks have been run.
- [ ] UI changes have been checked at desktop and mobile widths.
- [ ] API or scientific behavior changes include an explanation and validation.
- [ ] Documentation has been updated when needed.
