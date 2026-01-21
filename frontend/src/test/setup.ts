import '@testing-library/jest-dom';

// Mock scrollIntoView which isn't available in jsdom
Element.prototype.scrollIntoView = () => {};
