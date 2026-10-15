/**
 * Test environment shims. happy-dom has no ElementInternals, which @material/web's form-associated
 * elements (buttons, fields) ask for when they are created; a stand-in that accepts the calls lets
 * components using them mount.
 */
if (typeof HTMLElement !== 'undefined' && !('attachInternals' in HTMLElement.prototype)) {
  Object.defineProperty(HTMLElement.prototype, 'attachInternals', {
    value() {
      return {
        setFormValue() {},
        setValidity() {},
        checkValidity: () => true,
        reportValidity: () => true,
        form: null,
        labels: [],
        validity: { valid: true },
        validationMessage: '',
        willValidate: false,
        states: new Set(),
      };
    },
  });
}
