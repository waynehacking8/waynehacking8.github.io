class Blog {
    constructor() {
        this.posts = [];
        this.currentCategory = 'all';
        this.blogGrid = document.querySelector('.blog-grid');
        this.modal = document.getElementById('postModal');
        
        this.init();
    }

    async init() {
        try {
            await this.loadPosts();
            this.setupEventListeners();
            this.displayPosts();
        } catch (error) {
            console.error('部落格初始化失敗:', error);
            this.showError('初始化失敗，請重新整理頁面');
        }
    }

    setupEventListeners() {
        // 分類按鈕事件
        document.querySelectorAll('.category').forEach(button => {
            button.addEventListener('click', (e) => {
                document.querySelectorAll('.category').forEach(btn => 
                    btn.classList.remove('active'));
                button.classList.add('active');
                this.currentCategory = button.dataset.category;
                this.displayPosts();
            });
        });

        // 模態框關閉事件
        window.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });

        // 鍵盤事件
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    async loadPosts() {
        try {
            const response = await fetch('./posts.json');
            this.posts = await response.json();
            this.displayPosts(this.posts);
        } catch (error) {
            console.error('Error loading posts:', error);
        }
    }

    async loadPostContent(postId) {
        const post = this.posts.find(p => p.id === postId);
        if (!post) return null;
        
        try {
            const response = await fetch(post.file);
            const content = await response.text();
            return content;
        } catch (error) {
            console.error('Error loading post content:', error);
            return null;
        }
    }

    displayPosts() {
        const filteredPosts = this.currentCategory === 'all'
            ? this.posts
            : this.posts.filter(post => post.category === this.currentCategory);

        if (filteredPosts.length === 0) {
            this.blogGrid.innerHTML = '<div class="no-posts">此分類目前沒有文章</div>';
            return;
        }

        this.blogGrid.innerHTML = filteredPosts
            .map(post => this.createPostCard(post))
            .join('');
    }

    createPostCard(post) {
        return `
            <article class="blog-post" data-id="${post.id}">
                <div class="post-content">
                    <div class="post-category">
                        <i class="fas ${this.getCategoryIcon(post.category)}"></i>
                        ${this.formatCategory(post.category)}
                    </div>
                    <h2>${post.title}</h2>
                    <div class="post-meta">
                        <span class="post-date">
                            <i class="far fa-calendar"></i>
                            ${new Date(post.date).toLocaleDateString('zh-TW')}
                        </span>
                        <div class="post-tags">
                            ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                        </div>
                    </div>
                    <div class="post-excerpt-container">
                        <p class="post-excerpt">${post.excerpt || this.truncateContent(post.content)}</p>
                    </div>
                    <div class="post-actions">
                        <a href="./post.html?id=${post.id}" class="read-more-btn">
                            <span class="btn-text">閱讀更多</span>
                            <i class="fas fa-arrow-right"></i>
                        </a>
                    </div>
                </div>
            </article>
        `;
    }

    truncateContent(content) {
        // 移除 Markdown 標記，只保留純文字
        const plainText = content.replace(/[#*`]/g, '')
                                .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
                                .replace(/\n/g, ' ')
                                .trim();
        return plainText.length > 150 ? plainText.substring(0, 150) + '...' : plainText;
    }

    closeModal() {
        this.modal.style.display = 'none';
        document.body.style.overflow = '';
    }

    formatCategory(category) {
        const categoryMap = {
            'machine-learning': '機器學習',
            'cybersecurity': '資安',
            'programming': '程式開發',
            'others': '其他'
        };
        return categoryMap[category] || category;
    }

    getCategoryIcon(category) {
        const icons = {
            'machine-learning': 'fa-brain',
            'cybersecurity': 'fa-shield-alt',
            'programming': 'fa-code',
            'others': 'fa-ellipsis-h'
        };
        return icons[category] || 'fa-file-alt';
    }

    showError(message) {
        this.blogGrid.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                ${message}
            </div>
        `;
    }
}

// 初始化部落格
let blog;
document.addEventListener('DOMContentLoaded', () => {
    blog = new Blog();
}); 