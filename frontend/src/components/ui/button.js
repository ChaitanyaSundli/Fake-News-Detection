import React from 'react';

export function Button({ children, className = '', ...props }) {
  return (
    <button
      className={`px-4 py-2 rounded-lg font-medium focus:outline-none transition-colors duration-200 ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
