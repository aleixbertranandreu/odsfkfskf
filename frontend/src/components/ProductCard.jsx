import { Link } from 'react-router-dom';
import './ProductCard.css';

const colorMap = {
    'Beige': '#d4b896',
    'Negro': '#1a1a1a',
    'Blanco': '#ffffff',
    'Azul Claro': '#a8c4e0',
    'Crudo': '#f5f0e8',
    'Gris': '#9e9e9e',
    'Marino': '#1e3a5f',
    'Verde Salvia': '#8fa38b',
    'Azul Medio': '#5b8cc4',
    'Camel': '#c4956a',
};

export default function ProductCard({ product, style }) {
    return (
        <Link to={`/producto/${product.id}`} className="product-card fade-in" style={style}>
            <div className="product-card-image">
                <img
                    src={product.images[0]}
                    alt={product.name}
                    loading="lazy"
                />
                {product.isNew && <span className="product-card-new">Nuevo</span>}
            </div>
            <div className="product-card-info">
                <p className="product-card-name">{product.name}</p>
                <p className="product-card-price">{product.price.toFixed(2)} EUR</p>
                <div className="product-card-colors">
                    {product.colors.map((c) => (
                        <span
                            key={c}
                            className="color-dot"
                            style={{ background: colorMap[c] || '#ccc' }}
                            title={c}
                        />
                    ))}
                </div>
            </div>
        </Link>
    );
}
