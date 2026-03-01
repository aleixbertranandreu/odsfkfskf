import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import ProductCard from '../components/ProductCard';
import products from '../data/products';
import './ProductDetail.css';

const colorMap = {
    'Beige': '#d4b896', 'Negro': '#1a1a1a', 'Blanco': '#ffffff',
    'Azul Claro': '#a8c4e0', 'Crudo': '#f5f0e8', 'Gris': '#9e9e9e',
    'Marino': '#1e3a5f', 'Verde Salvia': '#8fa38b', 'Azul Medio': '#5b8cc4', 'Camel': '#c4956a',
};

export default function ProductDetail() {
    const { id } = useParams();
    const { addItem } = useCart();
    const [selectedSize, setSelectedSize] = useState('');
    const [added, setAdded] = useState(false);

    const product = products.find((p) => p.id === id);

    if (!product) {
        return (
            <main className="product-page">
                <div className="catalog-empty">
                    <h3>Producto no encontrado</h3>
                    <p>El producto que buscas no existe o ha sido retirado.</p>
                    <Link to="/catalogo" className="btn btn-primary" style={{ marginTop: '1.5rem', display: 'inline-flex' }}>
                        Ver catálogo
                    </Link>
                </div>
            </main>
        );
    }

    const related = products
        .filter((p) => p.category === product.category && p.id !== product.id)
        .slice(0, 4);

    const handleAdd = () => {
        if (!selectedSize) return;
        addItem(product, selectedSize);
        setAdded(true);
        setTimeout(() => setAdded(false), 2500);
    };

    return (
        <main className="product-page">
            <div className="product-layout fade-in">
                {/* Gallery */}
                <div className="product-gallery">
                    <div className="product-main-image">
                        <img src={product.images[0]} alt={product.name} />
                    </div>
                    {product.images.length > 1 && (
                        <div className="product-thumbs">
                            {product.images.map((img, i) => (
                                <div key={i} className={`product-thumb ${i === 0 ? 'active' : ''}`}>
                                    <img src={img} alt={`${product.name} ${i + 1}`} />
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Info */}
                <div className="product-info">
                    <div className="product-breadcrumb">
                        <Link to="/">Inicio</Link>
                        <span>/</span>
                        <Link to={`/catalogo?cat=${product.category}`}>
                            {product.category.charAt(0).toUpperCase() + product.category.slice(1)}
                        </Link>
                        <span>/</span>
                        {product.subcategory}
                    </div>

                    <h1 className="product-name">{product.name}</h1>
                    <p className="product-price">{product.price.toFixed(2)} EUR</p>
                    <p className="product-ref">Ref. {product.id}</p>

                    {/* Colors */}
                    <p className="product-colors-label">Color</p>
                    <div className="product-colors">
                        {product.colors.map((c) => (
                            <span
                                key={c}
                                className="color-dot"
                                style={{
                                    background: colorMap[c] || '#ccc',
                                    width: 24,
                                    height: 24,
                                    borderRadius: '50%',
                                    border: '2px solid var(--color-border)',
                                }}
                                title={c}
                            />
                        ))}
                    </div>

                    {/* Sizes */}
                    <p className="product-sizes-label">Talla</p>
                    <div className="product-sizes">
                        {product.sizes.map((s) => (
                            <button
                                key={s}
                                className={`size-btn ${selectedSize === s ? 'active' : ''}`}
                                onClick={() => setSelectedSize(s)}
                                aria-label={`Talla ${s}`}
                            >
                                {s}
                            </button>
                        ))}
                    </div>

                    {/* Description */}
                    <p className="product-desc">{product.description}</p>

                    {/* Add to cart */}
                    <button
                        className="product-add-btn"
                        onClick={handleAdd}
                        disabled={!selectedSize}
                    >
                        {selectedSize ? 'Añadir al carrito' : 'Selecciona una talla'}
                    </button>

                    {added && (
                        <p className="product-added-msg">✓ Artículo añadido al carrito</p>
                    )}
                </div>
            </div>

            {/* Related */}
            {related.length > 0 && (
                <section className="product-related">
                    <h2>También te puede interesar</h2>
                    <div className="related-grid">
                        {related.map((p, i) => (
                            <ProductCard key={p.id} product={p} style={{ animationDelay: `${i * 80}ms` }} />
                        ))}
                    </div>
                </section>
            )}
        </main>
    );
}
