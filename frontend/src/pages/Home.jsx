import { Link } from 'react-router-dom';
import ProductCard from '../components/ProductCard';
import products from '../data/products';
import './Home.css';

const categories = [
    { label: 'Mujer', query: 'mujer', img: '/images/black-dress.png' },
    { label: 'Hombre', query: 'hombre', img: '/images/leather-jacket.png' },
    { label: 'Kids', query: 'kids', img: '/images/knit-sweater.png' },
    { label: 'Novedades', query: 'novedades', img: '/images/camel-coat.png' },
];

export default function Home({ onScanOpen }) {
    const newArrivals = products.filter((p) => p.isNew).slice(0, 4);

    return (
        <main>
            {/* Hero */}
            <section className="home-hero">
                <img src="/images/hero.png" alt="Colección Primavera 2026" />
                <div className="home-hero-overlay" />
                <div className="home-hero-content fade-in">
                    <h1>Primavera 2026</h1>
                    <p>La nueva colección ya está aquí</p>
                    <Link to="/catalogo?cat=novedades" className="btn btn-secondary">
                        Ver colección
                    </Link>
                </div>
            </section>

            {/* Categories */}
            <section className="home-categories">
                <div className="home-section-title">
                    <h2>Explorar</h2>
                    <p>Descubre nuestras colecciones</p>
                </div>
                <div className="categories-grid">
                    {categories.map((cat) => (
                        <Link to={`/catalogo?cat=${cat.query}`} key={cat.query} className="category-card">
                            <img src={cat.img} alt={cat.label} loading="lazy" />
                            <div className="category-card-overlay" />
                            <span className="category-card-label">{cat.label}</span>
                        </Link>
                    ))}
                </div>
            </section>

            {/* New Arrivals */}
            <section className="home-arrivals">
                <div className="home-section-title">
                    <h2>Novedades</h2>
                    <p>Lo último en llegar a nuestra tienda</p>
                </div>
                <div className="arrivals-grid">
                    {newArrivals.map((p, i) => (
                        <ProductCard key={p.id} product={p} style={{ animationDelay: `${i * 100}ms` }} />
                    ))}
                </div>
            </section>

            {/* Scan Banner */}
            <section className="home-scan-banner">
                <div className="scan-banner-inner">
                    <div className="scan-banner-text">
                        <h2>¿Te gusta una prenda?</h2>
                        <p>
                            Sube una foto de cualquier prenda que te inspire y nuestro sistema
                            encontrará la más parecida en nuestro catálogo. Rápido, sencillo y preciso.
                        </p>
                    </div>
                    <button className="btn btn-secondary" onClick={onScanOpen}>
                        ✦ Escanear prenda
                    </button>
                </div>
            </section>
        </main>
    );
}
