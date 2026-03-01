import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import './Navbar.css';

export default function Navbar({ onScanOpen }) {
    const [scrolled, setScrolled] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);
    const { totalItems } = useCart();
    const location = useLocation();

    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 10);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    useEffect(() => {
        setMobileOpen(false);
    }, [location]);

    const navLinks = [
        { to: '/catalogo?cat=novedades', label: 'Novedades' },
        { to: '/catalogo?cat=mujer', label: 'Mujer' },
        { to: '/catalogo?cat=hombre', label: 'Hombre' },
        { to: '/catalogo?cat=kids', label: 'Kids' },
    ];

    return (
        <>
            <nav className={`navbar ${scrolled ? 'scrolled' : ''}`} role="navigation" aria-label="Navegación principal">
                <div className="navbar-inner">
                    <div className="navbar-left">
                        <Link to="/" className="navbar-logo" aria-label="Ir al inicio">
                            MODÄ
                        </Link>
                        <div className="navbar-links">
                            {navLinks.map((link) => (
                                <Link key={link.to} to={link.to} className={location.search.includes(link.label.toLowerCase()) ? 'active' : ''}>
                                    {link.label}
                                </Link>
                            ))}
                        </div>
                    </div>

                    <div className="navbar-right">
                        <button className="scan-cta" onClick={onScanOpen} aria-label="Escanear prenda">
                            ✦ Escanear Prenda
                        </button>

                        <Link to="/catalogo" className="navbar-icon-btn" aria-label="Buscar">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <circle cx="11" cy="11" r="8" />
                                <path d="m21 21-4.35-4.35" />
                            </svg>
                        </Link>

                        <Link to="/carrito" className="navbar-icon-btn" aria-label={`Carrito (${totalItems} artículos)`}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z" />
                                <line x1="3" y1="6" x2="21" y2="6" />
                                <path d="M16 10a4 4 0 01-8 0" />
                            </svg>
                            {totalItems > 0 && <span className="cart-badge">{totalItems}</span>}
                        </Link>

                        <button
                            className={`hamburger ${mobileOpen ? 'open' : ''}`}
                            onClick={() => setMobileOpen(!mobileOpen)}
                            aria-label="Menú"
                            aria-expanded={mobileOpen}
                        >
                            <span />
                            <span />
                            <span />
                        </button>
                    </div>
                </div>
            </nav>

            <div className={`mobile-menu ${mobileOpen ? 'open' : ''}`}>
                {navLinks.map((link) => (
                    <Link key={link.to} to={link.to}>
                        {link.label}
                    </Link>
                ))}
                <button onClick={() => { onScanOpen(); setMobileOpen(false); }}>
                    Escanear Prenda
                </button>
            </div>

            {/* Spacer */}
            <div style={{ height: 'var(--nav-height)' }} />
        </>
    );
}
