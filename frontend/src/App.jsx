import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { CartProvider } from './context/CartContext';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ScanModal from './components/ScanModal';
import Home from './pages/Home';
import Catalog from './pages/Catalog';
import ProductDetail from './pages/ProductDetail';
import Cart from './pages/Cart';
import Checkout from './pages/Checkout';

export default function App() {
  const [scanOpen, setScanOpen] = useState(false);

  return (
    <BrowserRouter>
      <CartProvider>
        <Navbar onScanOpen={() => setScanOpen(true)} />
        <Routes>
          <Route path="/" element={<Home onScanOpen={() => setScanOpen(true)} />} />
          <Route path="/catalogo" element={<Catalog />} />
          <Route path="/producto/:id" element={<ProductDetail />} />
          <Route path="/carrito" element={<Cart />} />
          <Route path="/checkout" element={<Checkout />} />
        </Routes>
        <Footer />
        <ScanModal isOpen={scanOpen} onClose={() => setScanOpen(false)} />
      </CartProvider>
    </BrowserRouter>
  );
}
