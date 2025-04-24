// src/components/ui/card.js
import React from 'react';

export const Card = ({ children }) => (
  <div
    style={{
      padding: '20px',
      border: '1px solid #ddd',
      borderRadius: '12px',
      boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
      backgroundColor: '#fff',
      marginBottom: '20px',
    }}
  >
    {children}
  </div>
);

export const CardContent = ({ children }) => (
  <div style={{ padding: '10px 0' }}>{children}</div>
);
