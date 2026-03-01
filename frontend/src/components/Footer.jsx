import './Footer.css';

export default function Footer() {
    return (
        <footer className="footer">
            <div className="footer-grid">
                <div className="footer-brand">
                    <h3>MODÄ</h3>
                    <p>
                        Moda contemporánea con un enfoque en la simplicidad, la calidad y la
                        expresión personal. Cada pieza cuenta una historia.
                    </p>
                    <div className="footer-newsletter">
                        <input type="email" placeholder="Tu email" aria-label="Email para newsletter" />
                        <button type="button">Suscribir</button>
                    </div>
                </div>

                <div className="footer-col">
                    <h4>Comprar</h4>
                    <a href="/catalogo?cat=mujer">Mujer</a>
                    <a href="/catalogo?cat=hombre">Hombre</a>
                    <a href="/catalogo?cat=kids">Kids</a>
                    <a href="/catalogo?cat=novedades">Novedades</a>
                </div>

                <div className="footer-col">
                    <h4>Ayuda</h4>
                    <a href="#">Atención al cliente</a>
                    <a href="#">Envíos</a>
                    <a href="#">Devoluciones</a>
                    <a href="#">Tallas</a>
                </div>

                <div className="footer-col">
                    <h4>Empresa</h4>
                    <a href="#">Sobre nosotros</a>
                    <a href="#">Sostenibilidad</a>
                    <a href="#">Tiendas</a>
                    <a href="#">Trabaja con nosotros</a>
                </div>
            </div>

            <div className="footer-bottom">
                <p>© 2026 MODÄ. Todos los derechos reservados.</p>
                <div className="footer-socials">
                    <a href="#" aria-label="Instagram">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <rect x="2" y="2" width="20" height="20" rx="5" ry="5" />
                            <path d="M16 11.37A4 4 0 1112.63 8 4 4 0 0116 11.37z" />
                            <line x1="17.5" y1="6.5" x2="17.51" y2="6.5" />
                        </svg>
                    </a>
                    <a href="#" aria-label="Twitter">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z" />
                        </svg>
                    </a>
                    <a href="#" aria-label="Pinterest">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M12 2C6.48 2 2 6.48 2 12c0 4.42 2.87 8.17 6.84 9.49-.09-.79-.18-2.01.04-2.87.2-.78 1.28-5.42 1.28-5.42s-.33-.65-.33-1.62c0-1.52.88-2.65 1.98-2.65.93 0 1.38.7 1.38 1.54 0 .94-.6 2.34-.91 3.64-.26 1.09.55 1.98 1.62 1.98 1.94 0 3.44-2.05 3.44-5.01 0-2.62-1.88-4.45-4.57-4.45-3.11 0-4.94 2.33-4.94 4.75 0 .94.36 1.95.81 2.5.09.11.1.21.08.32l-.3 1.24c-.05.2-.16.24-.37.14C6.93 14.29 5.9 12.33 5.9 11.7c0-3.31 2.41-6.36 6.94-6.36 3.64 0 6.48 2.6 6.48 6.07 0 3.62-2.28 6.53-5.45 6.53-1.06 0-2.07-.55-2.41-1.21l-.66 2.5c-.24.92-.88 2.06-1.31 2.76.99.3 2.03.47 3.11.47 5.52 0 10-4.48 10-10S17.52 2 12 2z" />
                        </svg>
                    </a>
                </div>
            </div>
        </footer>
    );
}
